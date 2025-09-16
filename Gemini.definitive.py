
"""
Gemini Live – videollamada con Python (audio + vídeo)

Este script crea una "videollamada" con la API Live de Gemini 2.5 Flash,
para que el modelo vea la cámara en vivo, escuche tu micrófono y te responda con voz.
"""

import asyncio
import os
import sys
import signal
import time
from typing import Optional

import numpy as np

# Dependencias opcionales (el script no debe crashear si faltan, solo desactiva funciones)
try:
    import sounddevice as sd  # Captura y reproducción de audio
except Exception as e:
    sd = None
    print("[ADVERTENCIA] sounddevice no disponible:", e, file=sys.stderr)

try:
    import cv2  # Webcam
except Exception as e:
    cv2 = None
    print("[ADVERTENCIA] OpenCV (cv2) no disponible:", e, file=sys.stderr)

try:
    from google import genai
    from google.genai import types
except Exception as e:
    genai = None
    types = None
    print("[ERROR] Falta google-genai. Instala con: pip install -U google-genai", file=sys.stderr)

API_KEY_ENV = "GEMINI_API_KEY"
MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-live-2.5-flash-preview")

# Audio
IN_RATE = 16000    # entrada micro
OUT_RATE = 24000   # salida TTS
CHANNELS = 1
IN_DTYPE = "int16"
OUT_DTYPE = "int16"

# Vídeo (ajustable por env)
FRAME_W = int(os.environ.get("GLIVE_FRAME_W", 320))
FRAME_H = int(os.environ.get("GLIVE_FRAME_H", 240))
FRAME_INTERVAL = float(os.environ.get("GLIVE_FRAME_INTERVAL", 0.2))  # ~5 fps
JPEG_QUALITY = int(os.environ.get("GLIVE_JPEG_QUALITY", 70))

# Reintentos
MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 3.0


class GracefulExit(SystemExit):
    """Señal de salida controlada."""
    pass


def _install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    def _raise_exit(*_):
        raise GracefulExit()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _raise_exit)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _raise_exit())


async def audio_capture_task(session, stop_event: asyncio.Event) -> None:
    """Lee del micrófono y envía audio PCM16 16 kHz al Live API."""
    if sd is None:
        print("[AUDIO] sounddevice no disponible; se omite captura.")
        await stop_event.wait()
        return

    blocksize = IN_RATE // 10  # 100 ms
    q: "asyncio.Queue[bytes]" = asyncio.Queue(maxsize=10)

    def _in_cb(indata, frames, time_info, status):
        if status:
            print(f"[AUDIO-IN][status] {status}", file=sys.stderr)
        try:
            q.put_nowait(bytes(indata))
        except asyncio.QueueFull:
            # Descartar para evitar bloqueo
            pass

    with sd.InputStream(samplerate=IN_RATE, channels=CHANNELS, dtype=IN_DTYPE,
                        blocksize=blocksize, callback=_in_cb):
        print("[AUDIO] Capturando micrófono…")
        try:
            while not stop_event.is_set():
                try:
                    chunk = await asyncio.wait_for(q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                try:
                    if types is not None:
                        await session.send_realtime_input(
                            audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={IN_RATE}")
                        )
                except Exception as e:
                    print(f"[AUDIO] Error enviando chunk: {e}", file=sys.stderr)
                    await asyncio.sleep(0.1)
        except GracefulExit:
            pass
        finally:
            try:
                await session.send_realtime_input(audio_stream_end=True)
            except Exception:
                pass
            print("[AUDIO] Captura detenida.")


async def audio_playback_task(session, stop_event: asyncio.Event) -> None:
    if sd is None:
        print("[AUDIO] sounddevice no disponible; no se reproducirá audio.")
        await stop_event.wait()
        return

    out_q: "asyncio.Queue[bytes]" = asyncio.Queue(maxsize=200)
    bytes_per_sample = 2  # int16
    bytes_per_frame = bytes_per_sample * CHANNELS
    buf = bytearray()

    def _out_fill(outdata, frames, time_info, status):
        nonlocal buf
        if status:
            print(f"[AUDIO-OUT][status] {status}", file=sys.stderr)

        bytes_per_sample = 2  # int16
        bytes_per_frame = bytes_per_sample * CHANNELS
        needed = frames * bytes_per_frame

        # Rellenar 'buf' con chunks de la cola hasta cubrir lo que el dispositivo pide
        while len(buf) < needed:
            try:
                chunk = out_q.get_nowait()
                buf.extend(chunk)
            except asyncio.QueueEmpty:
                break

        if len(buf) >= needed:
            # ¡IMPORTANTE! Copiar a 'bytes' para que NumPy no referencie 'buf'
            chunk = bytes(buf[:needed])
            del buf[:needed]
            arr = np.frombuffer(chunk, dtype=np.int16)
            outdata[:] = arr.reshape(-1, CHANNELS)
        else:
            # Infra-llenado: reproducimos lo disponible y rellenamos con ceros
            if len(buf) > 0:
                chunk = bytes(buf)  # copiamos lo que haya
                buf.clear()  # ya lo hemos consumido
                part = np.frombuffer(chunk, dtype=np.int16)
                tmp = np.zeros(frames * CHANNELS, dtype=np.int16)
                n = min(tmp.size, part.size)
                tmp[:n] = part[:n]
                outdata[:] = tmp.reshape(-1, CHANNELS)
            else:
                outdata[:] = np.zeros((frames, CHANNELS), dtype=np.int16)

    with sd.OutputStream(
        samplerate=OUT_RATE, channels=CHANNELS, dtype=OUT_DTYPE,
        blocksize=OUT_RATE // 20,  # ~50 ms está bien; puedes probar //40 (~25 ms)
        callback=_out_fill
    ):
        print("[AUDIO] Reproduciendo respuestas…")

        # Buffers de transcripción
        in_text_parts, out_text_parts = [], []

        while not stop_event.is_set():
            try:
                async for message in session.receive():
                    # Audio del modelo -> a la cola (sin perder sobrantes)
                    if getattr(message, "data", None):
                        try:
                            out_q.put_nowait(message.data)
                        except asyncio.QueueFull:
                            # Evita latencia infinita: suelta el chunk más viejo
                            try: _ = out_q.get_nowait()
                            except asyncio.QueueEmpty: pass
                            try: out_q.put_nowait(message.data)
                            except Exception: pass

                    sc = getattr(message, "server_content", None)

                    # Transcripciones: acumula, NO imprimas trocitos
                    if sc and getattr(sc, "input_transcription", None):
                        in_text_parts.append(sc.input_transcription.text)
                    if sc and getattr(sc, "output_transcription", None):
                        out_text_parts.append(sc.output_transcription.text)

                    # Fin de actividad del usuario -> vuelca input
                    if getattr(message, "activity_end", False) or getattr(message, "activityEnd", False):
                        if in_text_parts:
                            print("\n[TÚ] ", "".join(in_text_parts))
                            in_text_parts.clear()

                    # Turno del modelo completo -> vuelca output (y limpia)
                    if sc and getattr(sc, "turn_complete", False):
                        if out_text_parts:
                            print("\n[IA] ", "".join(out_text_parts))
                            out_text_parts.clear()

                await asyncio.sleep(0.05)
            except Exception as e:
                print(f"[AUDIO] Error recibiendo audio: {e}", file=sys.stderr)
                await asyncio.sleep(0.2)

        print("[AUDIO] Recepción detenida.")


async def video_capture_task(session, stop_event: asyncio.Event) -> None:
    """Lee la webcam con baja latencia, codifica a JPEG y envía SIEMPRE el frame más reciente."""
    if cv2 is None:
        print("[VIDEO] OpenCV no disponible; se omite vídeo.")
        await stop_event.wait()
        return

    # En Windows, CAP_DSHOW suele dar menos latencia que MSMF
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[VIDEO] No se pudo abrir la cámara. Continuando sin vídeo.", file=sys.stderr)
        await stop_event.wait()
        return

    # Preferencias de baja latencia (algunas cámaras/driver pueden ignorarlas, no pasa nada)
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception: pass
    try: cap.set(cv2.CAP_PROP_FPS, 30)
    except Exception: pass
    try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # muchas webcams van más sueltas con MJPG
    except Exception: pass

    last_sent = 0.0
    print("[VIDEO] Enviando frames… (Q para salir)")

    try:
        while not stop_event.is_set():
            # --- DRENA EL BÚFER DE LA CÁMARA ---
            # Saltamos varios frames en cola para quedarnos con el MÁS RECIENTE
            # (límite temporal ~10 ms para no comernos CPU).
            t0 = time.time()
            grabs = 0
            while True:
                ok = cap.grab()
                if not ok:
                    break
                grabs += 1
                if grabs >= 6 or (time.time() - t0) > 0.01:
                    break

            # Recupera el último frame recién “grabbed”
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                await asyncio.sleep(0.01)
                continue

            # Mostrar / tecla Q (opcional)
            try:
                disp = cv2.resize(frame, (FRAME_W, FRAME_H))
                cv2.imshow("Gemini Live – Cámara", disp)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    stop_event.set()
                    break
            except Exception:
                pass

            # Throttle de envío (usa GLIVE_FRAME_INTERVAL para ajustarlo)
            now = time.time()
            if now - last_sent < FRAME_INTERVAL:
                await asyncio.sleep(0.001)
                continue

            # Codifica y envía el ÚLTIMO frame (no acumulamos cola)
            try:
                resized = cv2.resize(frame, (FRAME_W, FRAME_H))
                ok, enc = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if not ok:
                    continue
                jpg_bytes = enc.tobytes()
                if types is not None:
                    await session.send_realtime_input(
                        video=types.Blob(data=jpg_bytes, mime_type="image/jpeg")
                    )
                last_sent = now
            except Exception as e:
                print(f"[VIDEO] Error enviando frame: {e}", file=sys.stderr)
                await asyncio.sleep(0.005)
    except GracefulExit:
        pass
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[VIDEO] Captura detenida.")


async def run_session_once(client) -> None:
    """Crea una sesión Live y lanza tareas concurrentes de audio (in/out) y vídeo."""
    # Determinar media_resolution con enum cuando esté disponible
    if types is not None and hasattr(types, "MediaResolution"):
        media_res = types.MediaResolution.MEDIA_RESOLUTION_LOW
    else:
        media_res = "MEDIA_RESOLUTION_LOW"

    config = {
        "response_modalities": ["AUDIO"],
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": os.environ.get("GLIVE_VOICE", "Kore")}},
            "language_code": os.environ.get("GLIVE_LANG", "es-ES"),
        },
        "media_resolution": media_res,
        "system_instruction": (
            "Eres un asistente por voz, conciso y amable. Describe en español lo que ves en la cámara cuando te lo pida. "
            "Si no tienes contexto visual suficiente, pide que acerquen el objeto o mejoren la iluminación."
        ),
        "input_audio_transcription": {},
        "output_audio_transcription": {},
    }

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        print(f"[LIVE] Sesión iniciada con modelo: {MODEL}")
        stop_event = asyncio.Event()

        tasks = [
            asyncio.create_task(audio_capture_task(session, stop_event), name="audio_in"),
            asyncio.create_task(audio_playback_task(session, stop_event), name="audio_out"),
            asyncio.create_task(video_capture_task(session, stop_event), name="video"),
        ]

        # Saludo inicial
        try:
            await session.send_client_content(
                turns={"role": "user", "parts": [{"text": "Hola, ¿me oyes?"}]},
                turn_complete=True,
            )
        except Exception:
            pass

        try:
            await stop_event.wait()
        except GracefulExit:
            pass
        finally:
            stop_event.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(0.1)
            print("[LIVE] Sesión finalizada.")


async def main() -> None:
    if genai is None:
        print("[ERROR] No se puede continuar sin google-genai. Instala con: pip install -U google-genai", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        print(f"[ERROR] Define la variable de entorno {API_KEY_ENV} con tu API key.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    loop = asyncio.get_event_loop()
    _install_signal_handlers(loop)

    retries = 0
    while True:
        try:
            await run_session_once(client)
            break
        except GracefulExit:
            break
        except Exception as e:
            retries += 1
            if retries > MAX_RETRIES:
                print(f"[ERROR] Sesión falló y se superó el máximo de reintentos: {e}", file=sys.stderr)
                break
            wait = RETRY_BACKOFF_SEC * retries
            print(f"[WARN] Error en sesión: {e}. Reintentando en {wait:.1f}s…", file=sys.stderr)
            await asyncio.sleep(wait)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except GracefulExit:
        print("\n[SALIDA] Señal de cierre recibida. Hasta pronto.")
    except KeyboardInterrupt:
        print("\n[SALIDA] Interrumpido por el usuario.")
