# -*- coding: utf-8 -*-
"""
Gemini AV (Audio/Video) – Live session con optimizaciones de latencia

Cambios clave frente a la versión base:
- JPEG con cv2.imencode (fallback a PIL) -> menos CPU y copias
- Búfer de cámara drenado + CAP_PROP_BUFFERSIZE=1 + MJPG -> menos lag
- Envíos a Live API con timeout + contador de errores -> evita bloqueos
- Jitter buffer de audio con batches ~40 ms y prebuffer configurable
- Drenaje de colas y filtros de payloads vacíos
"""

import os
import cv2
import time
import base64
import asyncio
import contextlib
import threading
import numpy as np
import sounddevice as sd
from io import BytesIO
from PIL import Image  # Se mantiene como fallback del encoder JPEG
from dotenv import load_dotenv

print("[init] Cargando dependencias...", flush=True)
try:
    from google import genai
except Exception as e:
    print("[error] No se pudo importar google.genai:", e, flush=True)
    raise

# ========= Config =========
IN_TARGET_RATE = 16000     # lo que espera el modelo (entrada mic -> 16 kHz)
OUT_MODEL_RATE = 24000     # Gemini devuelve audio a 24 kHz
CHANNELS = 1
FRAME_INTERVAL = float(os.getenv("GLIVE_FRAME_INTERVAL", "1.5"))  # s – enviar 1 frame / 1.5 s por defecto
CAM_INDEX = 0
WIN_NAME = "Gemini AV (Q para salir)"
SEND_TIMEOUT_SEC = float(os.getenv("GLIVE_SEND_TIMEOUT_SEC", "2.0"))
MAX_SEND_ERRORS = int(os.getenv("GLIVE_MAX_SEND_ERRORS", "8"))

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
GEMINI_API_KEY_ENV = os.getenv("GEMINI_API_KEY_ENV", "GEMINI_API_KEY")


# ======== Utils ========
def _b64(data: bytes) -> str:
    """Base64 sin copias extra."""
    return base64.b64encode(data).decode("utf-8")


def encode_audio_int16_pcm(data_int16: np.ndarray) -> dict:
    """Empaqueta PCM int16 mono para el endpoint Live."""
    if data_int16.size == 0:
        payload = b""
    else:
        payload = data_int16.tobytes() if data_int16.flags["C_CONTIGUOUS"] else np.ascontiguousarray(data_int16).tobytes()
    return {
        "mime_type": "audio/pcm",
        "data": _b64(payload),
    }


def encode_image_b64_jpeg(frame_bgr: np.ndarray, quality: int = 85) -> dict:
    """Codifica BGR (OpenCV) a JPEG en base64. Usa cv2.imencode; si falla, recurre a PIL."""
    try:
        ok, enc = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if ok:
            return {"mime_type": "image/jpeg", "data": _b64(enc.tobytes())}
        # Fallback a PIL si no se pudo encodar con OpenCV
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        with BytesIO() as bio:
            pil_image.save(bio, format="JPEG", quality=quality)
            b = bio.getvalue()
        return {"mime_type": "image/jpeg", "data": _b64(b)}
    except Exception:
        # Último recurso: PIL
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        with BytesIO() as bio:
            pil_image.save(bio, format="JPEG", quality=quality)
            b = bio.getvalue()
        return {"mime_type": "image/jpeg", "data": _b64(b)}


def resample_int16(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Reescala por interpolación lineal rápida (suficiente para voz)."""
    if src_rate == dst_rate or x.size == 0:
        return x
    ratio = dst_rate / float(src_rate)
    n_out = int(round(x.size * ratio))
    if n_out <= 0:
        return np.empty((0,), dtype=np.int16)
    # Interpolación lineal
    xp = np.linspace(0, 1, x.size, endpoint=False, dtype=np.float64)
    fp = x.astype(np.float64)
    x_new = np.linspace(0, 1, n_out, endpoint=False, dtype=np.float64)
    y = np.interp(x_new, xp, fp)
    y = np.clip(y, -32768, 32767).astype(np.int16)
    return y


# ======== Cliente Live ========
class GeminiLiveClient:
    def __init__(self, api_key: str, model: str = GEMINI_MODEL):
        self.api_key = api_key
        self.model = model
        self.session = None
        self.session_cm = None
        self.quit = asyncio.Event()
        self.rx_audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._last_img_ts = 0.0
        self._send_errors = 0

    async def connect(self):
        print("[gemini] Conectando sesión live...", flush=True)
        client = genai.Client(api_key=self.api_key, http_options={"api_version": "v1alpha"})
        cfg = {"response_modalities": ["AUDIO"]}
        self.session_cm = client.aio.live.connect(model=self.model, config=cfg)
        self.session = await self.session_cm.__aenter__()
        print("[gemini] Conectado.", flush=True)
        asyncio.create_task(self._receiver_loop())

    async def close(self):
        self.quit.set()
        if self.session_cm is not None:
            with contextlib.suppress(Exception):
                await self.session_cm.__aexit__(None, None, None)
        self.session = None
        self.session_cm = None
        print("[gemini] Sesión cerrada.", flush=True)

    async def _receiver_loop(self):
        print("[gemini] Receiver loop iniciado.", flush=True)
        try:
            while not self.quit.is_set():
                turn = self.session.receive()
                async for response in turn:
                    data = getattr(response, "data", None)
                    if data:
                        audio = np.frombuffer(data, dtype=np.int16)
                        if audio.size > 0:
                            await self.rx_audio_queue.put(audio)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print("[gemini] Receiver loop terminó (info/ok al cerrar):", e, flush=True)
        finally:
            print("[gemini] Receiver loop finalizado.", flush=True)

    async def _send_input(self, payload: dict):
        """Envía input con timeout y control de errores acumulados."""
        try:
            await asyncio.wait_for(self.session.send(input=payload), timeout=SEND_TIMEOUT_SEC)  # API histórica estable
            self._send_errors = 0  # reset al tener éxito
        except asyncio.TimeoutError:
            self._send_errors += 1
            if self._send_errors >= MAX_SEND_ERRORS:
                print("[gemini] Demasiados timeouts enviando; solicitando cierre suave.", flush=True)
                self.quit.set()
        except Exception as e:
            self._send_errors += 1
            if self._send_errors % 3 == 1:
                print(f"[gemini] Aviso al enviar (intento {self._send_errors}): {e}", flush=True)
            if self._send_errors >= MAX_SEND_ERRORS:
                print("[gemini] Acumulación de errores de envío; solicitando cierre suave.", flush=True)
                self.quit.set()

    async def send_audio_int16(self, pcm_int16: np.ndarray):
        if self.session is None or pcm_int16.size == 0:
            return
        await self._send_input(encode_audio_int16_pcm(pcm_int16))

    async def maybe_send_frame(self, frame_bgr: np.ndarray, force: bool = False):
        if self.session is None:
            return
        now = time.time()
        if force or (now - self._last_img_ts) >= FRAME_INTERVAL:
            self._last_img_ts = now
            await self._send_input(encode_image_b64_jpeg(frame_bgr))


# ======== MIC: productor con backpressure + resample ========
async def mic_producer(loop, audio_out_q: asyncio.Queue[np.ndarray]):
    """Captura del micrófono en la tasa nativa del dispositivo y re-muestreo a 16 kHz mono int16."""
    try:
        dev = sd.query_devices(kind='input')
        in_native_rate = int(round(dev['default_samplerate']))
    except Exception:
        in_native_rate = IN_TARGET_RATE

    blocksize = max(128, int(in_native_rate * 0.06))  # ~60 ms por bloque
    q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=16)

    def callback(indata, frames, time_info, status):
        if status:
            # No spamear; status puede llegar a menudo
            if status.input_overflow:
                print("[audio] overflow entrada", flush=True)
        try:
            q.put_nowait(bytes(indata))
        except asyncio.QueueFull:
            # drop oldest
            try:
                _ = q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            with contextlib.suppress(Exception):
                q.put_nowait(bytes(indata))

    with sd.RawInputStream(samplerate=in_native_rate, channels=CHANNELS,
                           dtype='int16', callback=callback, blocksize=blocksize):
        print(f"[audio] Mic listo (blocksize={blocksize}, rate={in_native_rate}).", flush=True)
        try:
            while True:
                b = await q.get()
                if not b:
                    continue
                pcm = np.frombuffer(b, dtype=np.int16)
                if in_native_rate != IN_TARGET_RATE:
                    pcm = resample_int16(pcm, in_native_rate, IN_TARGET_RATE)
                await audio_out_q.put(pcm)
        except asyncio.CancelledError:
            pass


# ======== ALTAVOZ con Jitter Buffer y Prebuffer ========
async def speaker_consumer(rx_audio_q: asyncio.Queue[np.ndarray], stop_evt: asyncio.Event):
    print("[audio] Preparando salida de altavoz con jitter buffer...", flush=True)
    try:
        dev = sd.query_devices(kind='output')
        out_rate = int(round(dev['default_samplerate']))
    except Exception:
        out_rate = OUT_MODEL_RATE

    print(f"[audio] Rate dispositivo salida: {out_rate} Hz (modelo: {OUT_MODEL_RATE} Hz)", flush=True)

    buffer = bytearray()
    buf_lock = threading.Lock()

    async def rx_feeder():
        # Reducimos latencia percibida agregando lotes más pequeños (≈40 ms)
        target_ms = int(os.getenv("GLIVE_RX_TARGET_MS", "40"))
        target_samples = max(1, int(OUT_MODEL_RATE * target_ms / 1000))
        acc = np.empty((0,), dtype=np.int16)
        try:
            while not stop_evt.is_set():
                try:
                    chunk = await asyncio.wait_for(rx_audio_q.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    chunk = None
                if chunk is not None and chunk.size > 0:
                    acc = np.concatenate((acc, chunk))
                    if acc.size >= target_samples:
                        out_i16 = resample_int16(acc, OUT_MODEL_RATE, out_rate)
                        with buf_lock:
                            buffer.extend(out_i16.tobytes())
                        acc = np.empty((0,), dtype=np.int16)
            if acc.size:
                out_i16 = resample_int16(acc, OUT_MODEL_RATE, out_rate)
                with buf_lock:
                    buffer.extend(out_i16.tobytes())
        except asyncio.CancelledError:
            pass

    feeder_task = asyncio.create_task(rx_feeder())

    # Prebuffer configurable; 200 ms reduce TTFB pero mantiene estabilidad
    pre_ms = int(os.getenv("GLIVE_PREBUFFER_MS", "200"))
    bytes_per_sec = out_rate * 2  # int16 mono
    pre_bytes = int(bytes_per_sec * pre_ms / 1000)
    print(f"[audio] Prebuffer objetivo: {pre_ms} ms ≈ {pre_bytes} bytes", flush=True)

    t0 = time.time()
    while not stop_evt.is_set():
        with buf_lock:
            blen = len(buffer)
        if blen >= pre_bytes:
            break
        await asyncio.sleep(0.01)
        if time.time() - t0 > 2.0:
            print("[audio] Aviso: no llegó audio en 2 s; iniciando de todos modos.", flush=True)
            break

    blocksize = max(64, int(out_rate * 0.05))
    print(f"[audio] Iniciando salida. blocksize={blocksize}", flush=True)

    def out_callback(outdata, frames, time_info, status):
        need_bytes = frames * 2  # int16 mono
        with buf_lock:
            if len(buffer) >= need_bytes:
                give = buffer[:need_bytes]
                del buffer[:need_bytes]
            else:
                give = bytes(buffer[:])
                buffer.clear()
        if len(give) < need_bytes:
            pad = (np.zeros((need_bytes - len(give)) // 2, dtype=np.int16)).tobytes()
            give += pad
        outdata[:] = give

    try:
        with sd.RawOutputStream(samplerate=out_rate, channels=CHANNELS,
                                dtype='int16', blocksize=blocksize,
                                callback=out_callback):
            print("[audio] Altavoz listo (jitter buffer activo).", flush=True)
            await stop_evt.wait()
    finally:
        print("[audio] Altavoz detenido. Cerrando feeder...", flush=True)
        feeder_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await feeder_task


# ======== CÁMARA ========
async def camera_loop(gem: GeminiLiveClient, stop_evt: asyncio.Event):
    print("[cam] Abriendo cámara...", flush=True)
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(CAM_INDEX)
    print(f"[cam] isOpened={cap.isOpened()}", flush=True)
    if not cap.isOpened():
        raise RuntimeError("No se puede abrir la cámara (prueba CAM_INDEX=1,2,3).")
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 960, 540)
    last_sent = 0.0
    print("[cam] Cámara lista. Ventana abierta. (Q para salir)", flush=True)

    # Baja latencia: minimizar buffering y preferir MJPG si se puede
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    try:
        while not stop_evt.is_set():
            # Drenar el búfer para tomar el frame más reciente y reducir lag
            grabs = 0
            t0 = time.time()
            while True:
                ok = cap.grab()
                if not ok:
                    break
                grabs += 1
                if grabs >= 5 or (time.time() - t0) > 0.01:
                    break
            ok, frame = cap.retrieve()
            if not ok:
                print("[cam] No se pudo leer frame; reintentando...", flush=True)
                await asyncio.sleep(0.05)
                continue

            cv2.imshow(WIN_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                stop_evt.set()
                break

            now = time.time()
            if now - last_sent >= FRAME_INTERVAL:
                last_sent = now
                await gem.maybe_send_frame(frame)

            await asyncio.sleep(0.001)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[cam] Cámara cerrada.", flush=True)


# ======== ENVÍO AUDIO: drena cola y compacta ========
async def audio_send_loop(gem: GeminiLiveClient, mic_q: asyncio.Queue[np.ndarray], stop_evt: asyncio.Event):
    print("[audio] Bucle de envío a Gemini (con drenaje) iniciado.", flush=True)
    try:
        while not stop_evt.is_set():
            try:
                block = await asyncio.wait_for(mic_q.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            if block.size == 0:
                continue
            parts = [block]
            # Drenaje de cola para consolidar bloques adyacentes -> menos send()
            while True:
                try:
                    parts.append(mic_q.get_nowait())
                except asyncio.QueueEmpty:
                    break
            merged = np.concatenate(parts) if len(parts) > 1 else block
            await gem.send_audio_int16(merged)
    except asyncio.CancelledError:
        pass
    finally:
        print("[audio] Bucle de envío detenido.", flush=True)


# ======== MAIN ========
async def main():
    load_dotenv()
    api_key = os.getenv(GEMINI_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Define la variable de entorno {GEMINI_API_KEY_ENV} con tu API key.")

    gem = GeminiLiveClient(api_key=api_key, model=GEMINI_MODEL)
    await gem.connect()

    stop_evt = asyncio.Event()
    mic_q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=64)

    mic_task = asyncio.create_task(mic_producer(asyncio.get_event_loop(), mic_q))
    speaker_task = asyncio.create_task(speaker_consumer(gem.rx_audio_queue, stop_evt))
    cam_task = asyncio.create_task(camera_loop(gem, stop_evt))
    send_task = asyncio.create_task(audio_send_loop(gem, mic_q, stop_evt))

    try:
        await cam_task
    finally:
        print("[main] Cerrando...", flush=True)
        stop_evt.set()
        for t in [mic_task, send_task, speaker_task]:
            t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(mic_task, send_task, speaker_task, return_exceptions=True)
        await gem.close()
        print("[main] Salida limpia.", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        import traceback
        print("[fatal] Excepción no controlada:", e, flush=True)
        traceback.print_exc()
