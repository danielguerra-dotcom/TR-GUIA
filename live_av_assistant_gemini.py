# -*- coding: utf-8 -*-
"""
How to run
----------
1) Install the primary stack (safe versions):
   pip install "google-genai>=0.3" "sounddevice>=0.4" "opencv-python>=4.9" "numpy>=1.26" \
               "scipy>=1.12" "librosa>=0.10" "python-dotenv>=1.0" "httpx>=0.27" "loguru>=0.7"

   (Optional fallbacks that the script auto-selects if primaries are missing:
     soundcard>=0.4  pyaudio>=0.2.14  imageio>=2.34  pillow>=10.3  av>=12.0)

2) Create a .env next to this script (or export an env var) with:
   GEMINI_API_KEY=your_api_key_here

3) Run:
   python gemini_live_duplex_refactor.py

Notes
-----
- The script auto-selects libraries per subsystem (Primary first, then one Substitute).
- Clean English logs indicate which branches/devices were selected.
- Press 'Q' in the preview window to quit gracefully.
- Design targets Gemini Live AUDIO-only (v1alpha), low-latency duplex audio, and periodic JPEG frames.
"""

# ==============================
# CONFIG (tune here)
# ==============================

# Audio timing
MIC_BLOCK_MS   = 200          # mic capture block (~150–250 ms target)
PLAY_BLOCK_MS  = 60           # fixed speaker write size (~60 ms blocks)
PREBUFFER_S    = 0.30         # jitter prebuffer before starting playback (~300 ms)

# Uplink processing
NOISE_GATE_DBFS   = -45.0     # gate below this RMS level -> silence
AGC_TARGET_DBFS   = -20.0     # target RMS for AGC
AGC_MAX_GAIN_DB   = +15.0     # clamp max gain
AGC_ATTACK_MS     = 30.0      # faster gain up
AGC_RELEASE_MS    = 300.0     # slower gain down

# Model sample rates
UPLINK_HZ      = 16000        # model expects 16 kHz mono int16 for input
DOWNLINK_HZ    = 24000        # model returns 24 kHz mono int16

# Camera
FRAME_INTERVAL_S = 1.5        # send 1 JPEG about every 1.5 s
JPEG_QUALITY     = 82         # ~80–85

# Queues / backpressure
MIC_QUEUE_CAP   = 8           # bounded to cap latency (~8 * 200 ms = 1.6 s max in-flight)
RX_QUEUE_CAP    = 64          # downlink chunks queue
SEND_TIMEOUT_S  = 2.0
MAX_SEND_ERRORS = 8

# Devices (None = default)
MIC_DEVICE_INDEX    = None
SPEAKER_DEVICE_INDEX= None
CAM_INDEX           = 0

# Gemini Live
GEMINI_MODEL   = "gemini-2.0-flash-exp"
ENV_API_KEY    = "GEMINI_API_KEY"
API_VERSION    = "v1alpha"
RESPONSE_MODALITIES = ["AUDIO"]

# ==============================
# Imports & library selection
# ==============================

import os, sys, math, time, base64, contextlib, threading, queue, asyncio, signal
from dataclasses import dataclass
from typing import Optional, Tuple, Deque
from collections import deque

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{extra[tag]}</cyan> | {message}")

# .env
try:
    from dotenv import load_dotenv  # primary
    load_dotenv()
    DOTENV_OK = True
except Exception:
    DOTENV_OK = False

# --- Google GenAI client (Primary) ---
GENAI_BACKEND = None
try:
    from google import genai as _genai   # primary official async client
    GENAI_BACKEND = "google-genai"
except Exception as e:
    GENAI_BACKEND = None

# --- Audio I/O (Primary + substitutes) ---
AUDIO_BACKEND = None
_sd = None; _sc = None; _pa = None
try:
    import sounddevice as _sd  # Primary
    AUDIO_BACKEND = "sounddevice"
except Exception:
    try:
        import soundcard as _sc  # Substitute #1
        AUDIO_BACKEND = "soundcard"
    except Exception:
        try:
            import pyaudio as _pa  # Substitute #2
            AUDIO_BACKEND = "pyaudio"
        except Exception:
            AUDIO_BACKEND = None

# --- Camera / JPEG (Primary + substitutes) ---
CAM_BACKEND = None
_cv2 = None; _imgio = None; _PIL = None
try:
    import cv2 as _cv2  # Primary
    CAM_BACKEND = "opencv"
except Exception:
    try:
        import imageio as _imgio  # Substitute encoder
        from PIL import Image as _PIL  # Substitute
        CAM_BACKEND = "imageio+pillow"
    except Exception:
        CAM_BACKEND = None

# --- Arrays & math
import numpy as np

# --- Resampling (Primary + substitutes) ---
RESAMP_BACKEND = None
_resample_poly = None
try:
    from scipy.signal import resample_poly as _resample_poly  # Primary
    RESAMP_BACKEND = "scipy.resample_poly"
except Exception:
    try:
        import librosa  # Substitute
        RESAMP_BACKEND = "librosa.resample"
    except Exception:
        RESAMP_BACKEND = "linear"  # custom fallback


# ==============================
# Helpers (logging tags)
# ==============================

def log_tagged(tag, level="info"):
    def _wrap(msg):
        logger.bind(tag=tag).__getattribute__(level)(msg)
    return _wrap

log_main    = log_tagged("MAIN")
log_gem     = log_tagged("GEMINI")
log_mic     = log_tagged("MIC")
log_spk     = log_tagged("SPEAKER")
log_cam     = log_tagged("CAMERA")
log_proc    = log_tagged("DSP")
log_sys     = log_tagged("SYS")

# ==============================
# Utility functions (b64, dBFS, RMS)
# ==============================

def to_b64(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")

def rms_int16(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    # avoid overflow with float64
    return float(np.sqrt(np.mean((x.astype(np.float64))**2)))

def dbfs_from_rms(r: float) -> float:
    if r <= 0.5:
        return -120.0
    return 20.0 * math.log10(r / 32768.0)

def rms_from_dbfs(dbfs: float) -> float:
    return (10.0 ** (dbfs / 20.0)) * 32768.0

# ==============================
# Resampling (abstracted backends)
# ==============================

def resample_i16(x: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
    """Resample int16 mono with selected backend."""
    if x.size == 0 or src_hz == dst_hz:
        return x.astype(np.int16, copy=False)
    # Convert to float32 in [-1,1] for high-quality SRC
    xf = x.astype(np.float32) / 32768.0
    if RESAMP_BACKEND == "scipy.resample_poly" and _resample_poly is not None:
        # Use rational factors reduced by gcd for numerical stability
        g = math.gcd(dst_hz, src_hz)
        up, down = dst_hz // g, src_hz // g
        yf = _resample_poly(xf, up, down)
    elif RESAMP_BACKEND == "librosa.resample":
        import librosa  # type: ignore
        yf = librosa.resample(xf, orig_sr=src_hz, target_sr=dst_hz, res_type="kaiser_best")
    else:
        # linear fallback (fast, ok for voice)
        n_out = int(round(len(xf) * dst_hz / float(src_hz)))
        if n_out <= 0:
            return np.zeros((0,), dtype=np.int16)
        xi = np.linspace(0.0, 1.0, num=len(xf), endpoint=False, dtype=np.float64)
        xo = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
        yf = np.interp(xo, xi, xf.astype(np.float64)).astype(np.float32)
    # clip and convert back to int16
    yi = np.clip(yf * 32768.0, -32768, 32767).astype(np.int16)
    return yi

# ==============================
# AGC (RMS-based with attack/release)
# ==============================

@dataclass
class AgcState:
    current_gain: float = 1.0

class RmsAgc:
    def __init__(self, target_dbfs: float, max_gain_db: float,
                 attack_ms: float, release_ms: float):
        self.target_rms = rms_from_dbfs(target_dbfs)
        self.max_gain = 10.0 ** (max_gain_db / 20.0)
        self.attack_ms = max(1e-3, attack_ms)
        self.release_ms = max(1e-3, release_ms)
        self.state = AgcState(1.0)

    def process(self, x: np.ndarray, sample_rate: int) -> np.ndarray:
        if x.size == 0:
            return x
        block_ms = 1000.0 * (len(x) / float(sample_rate))
        # instantaneous desired gain
        r = rms_int16(x)
        if r < 1.0:  # avoid division by zero, keep near silence
            desired = 1.0
        else:
            desired = min(self.max_gain, self.target_rms / r)
        # smoothing per attack/release times (exponential approach)
        if desired > self.state.current_gain:
            tau = self.attack_ms
        else:
            tau = self.release_ms
        # compute smoothing factor for this block
        alpha = 1.0 - math.exp(-block_ms / tau)
        g = (1.0 - alpha) * self.state.current_gain + alpha * desired
        self.state.current_gain = g
        y = np.clip(x.astype(np.float64) * g, -32768, 32767).astype(np.int16)
        return y

# ==============================
# Noise gate
# ==============================

def noise_gate(x: np.ndarray, threshold_dbfs: float) -> np.ndarray:
    thr = rms_from_dbfs(threshold_dbfs)
    if rms_int16(x) < thr:
        return np.zeros_like(x)
    return x

# ==============================
# Gemini Live (async) wrapper
# ==============================

class LiveAIOBridge:
    def __init__(self, api_key: str, model: str):
        if GENAI_BACKEND != "google-genai":
            raise RuntimeError("Google GenAI client not available; please install 'google-genai>=0.3'.")
        self._client = _genai.Client(api_key=api_key, http_options={"api_version": API_VERSION})
        self._session_cm = None
        self.session = None
        self.rx_audio_q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=RX_QUEUE_CAP)
        self._rx_task: Optional[asyncio.Task] = None
        self._send_errors = 0
        self._closed = asyncio.Event()
        self._model = model

    async def open(self):
        log_gem(f"Connecting Live session (v{API_VERSION}, modalities={RESPONSE_MODALITIES})")
        cfg = {"response_modalities": RESPONSE_MODALITIES}
        self._session_cm = self._client.aio.live.connect(model=self._model, config=cfg)
        self.session = await self._session_cm.__aenter__()
        log_gem("Connected.")
        self._rx_task = asyncio.create_task(self._receiver())

    async def _receiver(self):
        log_gem("Receiver started.")
        try:
            while not self._closed.is_set():
                turn = self.session.receive()
                async for msg in turn:
                    data = getattr(msg, "data", None)
                    if data:
                        try:
                            chunk = np.frombuffer(data, dtype=np.int16)
                            if chunk.size:
                                try:
                                    self.rx_audio_q.put_nowait(chunk)
                                except asyncio.QueueFull:
                                    # Drop oldest to keep latency bounded
                                    with contextlib.suppress(asyncio.QueueEmpty):
                                        _ = self.rx_audio_q.get_nowait()
                                    self.rx_audio_q.put_nowait(chunk)
                        except Exception as e:
                            log_gem(f"Bad audio chunk: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log_gem(f"Receiver ended: {e}")
        finally:
            log_gem("Receiver stopped.")

    async def _send(self, payload: dict):
        try:
            await asyncio.wait_for(self.session.send(input=payload), timeout=SEND_TIMEOUT_S)
            self._send_errors = 0
        except asyncio.TimeoutError:
            self._send_errors += 1
            log_gem(f"Send timeout ({self._send_errors}/{MAX_SEND_ERRORS})")
            if self._send_errors >= MAX_SEND_ERRORS:
                log_gem("Too many send timeouts; requesting shutdown.")
                await self.close()
        except Exception as e:
            self._send_errors += 1
            if self._send_errors % 3 == 1:
                log_gem(f"Send error ({self._send_errors}): {e}")
            if self._send_errors >= MAX_SEND_ERRORS:
                log_gem("Accumulated send errors; requesting shutdown.")
                await self.close()

    async def send_pcm16(self, pcm: np.ndarray):
        if pcm.size == 0:
            return
        payload = {"mime_type": "audio/pcm", "data": to_b64(pcm.tobytes(order="C"))}
        await self._send(payload)

    async def send_jpeg_bgr(self, frame_bgr: np.ndarray):
        if CAM_BACKEND == "opencv" and _cv2 is not None:
            ok, enc = _cv2.imencode(".jpg", frame_bgr, [int(_cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
            if not ok:
                return
            b = enc.tobytes()
            payload = {"mime_type": "image/jpeg", "data": to_b64(b)}
            await self._send(payload)
        else:
            # Fallback: imageio + pillow path (convert BGR->RGB)
            if _PIL is None:
                return
            rgb = frame_bgr[..., ::-1]
            from PIL import Image
            from io import BytesIO
            img = Image.fromarray(rgb)
            with BytesIO() as bio:
                img.save(bio, format="JPEG", quality=JPEG_QUALITY)
                payload = {"mime_type": "image/jpeg", "data": to_b64(bio.getvalue())}
            await self._send(payload)

    async def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        if self._rx_task:
            self._rx_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._rx_task
        if self._session_cm is not None:
            with contextlib.suppress(Exception):
                await self._session_cm.__aexit__(None, None, None)
        log_gem("Session closed.")

# ==============================
# Mic capture thread -> asyncio queue
# ==============================

class MicThread(threading.Thread):
    """
    Captures mono int16 at the device native input rate in ~200 ms blocks.
    Pushes raw NumPy int16 blocks (and their native rate) into an asyncio.Queue
    using loop.call_soon_threadsafe; drops oldest if the queue is full.
    """
    daemon = True

    def __init__(self, loop: asyncio.AbstractEventLoop, out_q: asyncio.Queue, device_index=None):
        super().__init__(name="MicThread")
        self.loop = loop
        self.out_q = out_q
        self.device_index = device_index
        self.quit = threading.Event()
        self.native_rate = None
        self._local_q: "queue.Queue[bytes]" = queue.Queue(maxsize=32)

    def _async_put_drop_oldest(self, item):
        # Runs inside event loop thread (scheduled via call_soon_threadsafe)
        if self.out_q.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self.out_q.get_nowait()
        self.out_q.put_nowait(item)

    def stop(self):
        self.quit.set()

    def run(self):
        log_mic(f"Audio backend selected: {AUDIO_BACKEND}")
        try:
            if AUDIO_BACKEND == "sounddevice" and _sd is not None:
                self._run_sounddevice()
            elif AUDIO_BACKEND == "soundcard" and _sc is not None:
                self._run_soundcard()
            elif AUDIO_BACKEND == "pyaudio" and _pa is not None:
                self._run_pyaudio()
            else:
                log_mic("No audio input backend available. Exiting mic thread.")
        except Exception as e:
            log_mic(f"Mic thread error: {e}")
        finally:
            log_mic("Mic thread stopped.")

    def _run_sounddevice(self):
        # Discover default samplerate
        try:
            dev = _sd.query_devices(self.device_index if self.device_index is not None else None, kind='input')
            self.native_rate = int(round(dev['default_samplerate']))
        except Exception:
            self.native_rate = UPLINK_HZ
        blocksize = max(128, int(self.native_rate * (MIC_BLOCK_MS / 1000.0)))
        log_mic(f"Opening input (device={self.device_index}, rate={self.native_rate} Hz, block={blocksize} frames)")
        def cb(indata, frames, time_info, status):
            if status and status.input_overflow:
                log_mic("Input overflow (sounddevice)")
            try:
                self._local_q.put_nowait(bytes(indata))
            except queue.Full:
                with contextlib.suppress(Exception):
                    _ = self._local_q.get_nowait()
                with contextlib.suppress(Exception):
                    self._local_q.put_nowait(bytes(indata))

        with _sd.RawInputStream(samplerate=self.native_rate, channels=1, dtype='int16',
                               blocksize=blocksize, device=self.device_index, callback=cb):
            log_mic("Mic stream ready.")
            while not self.quit.is_set():
                try:
                    b = self._local_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                pcm = np.frombuffer(b, dtype=np.int16)
                item = (pcm, self.native_rate)
                self.loop.call_soon_threadsafe(self._async_put_drop_oldest, item)

    def _run_soundcard(self):
        mic = _sc.default_microphone() if self.device_index is None else _sc.all_microphones()[self.device_index]
        # soundcard prefers blocking recordings in float32; we'll convert to int16
        with mic.recorder(samplerate=None, channels=1) as rec:  # None -> native rate
            self.native_rate = int(rec.samplerate)
            block_frames = max(128, int(self.native_rate * (MIC_BLOCK_MS / 1000.0)))
            log_mic(f"Opening input (soundcard, rate={self.native_rate} Hz, block={block_frames} frames)")
            while not self.quit.is_set():
                data = rec.record(numframes=block_frames)  # float32 [-1,1]
                pcm = np.clip(data[:,0] * 32768.0, -32768, 32767).astype(np.int16, copy=False)
                item = (pcm, self.native_rate)
                self.loop.call_soon_threadsafe(self._async_put_drop_oldest, item)

    def _run_pyaudio(self):
        pa = _pa.PyAudio()
        try:
            # get default device info for input
            if self.device_index is None:
                self.device_index = pa.get_default_input_device_info().get('index', 0)
            dev_info = pa.get_device_info_by_index(self.device_index)
            self.native_rate = int(dev_info.get('defaultSampleRate', UPLINK_HZ))
            blocksize = max(128, int(self.native_rate * (MIC_BLOCK_MS / 1000.0)))
            log_mic(f"Opening input (pyaudio, device={self.device_index}, rate={self.native_rate} Hz, block={blocksize} frames)")
            stream = pa.open(format=_pa.paInt16, channels=1, rate=self.native_rate, input=True,
                             frames_per_buffer=blocksize, input_device_index=self.device_index)
            try:
                while not self.quit.is_set():
                    b = stream.read(blocksize, exception_on_overflow=False)
                    pcm = np.frombuffer(b, dtype=np.int16)
                    item = (pcm, self.native_rate)
                    self.loop.call_soon_threadsafe(self._async_put_drop_oldest, item)
            finally:
                stream.stop_stream(); stream.close()
        finally:
            pa.terminate()

# ==============================
# Speaker playback (async) with jitter buffer
# ==============================

async def speaker_player(rx_audio_q: asyncio.Queue, stop_evt: asyncio.Event):
    # Determine output device samplerate
    out_rate = None
    if AUDIO_BACKEND == "sounddevice" and _sd is not None:
        try:
            dev = _sd.query_devices(SPEAKER_DEVICE_INDEX if SPEAKER_DEVICE_INDEX is not None else None, kind='output')
            out_rate = int(round(dev['default_samplerate']))
        except Exception:
            out_rate = DOWNLINK_HZ
    elif AUDIO_BACKEND == "soundcard" and _sc is not None:
        spk = _sc.default_speaker() if SPEAKER_DEVICE_INDEX is None else _sc.all_speakers()[SPEAKER_DEVICE_INDEX]
        out_rate = int(spk.samplerate)
    else:
        # pyaudio speaker path not implemented for brevity; we require sounddevice/soundcard for output
        pass
    if out_rate is None:
        out_rate = DOWNLINK_HZ

    log_spk(f"Output device rate: {out_rate} Hz (model: {DOWNLINK_HZ} Hz)")
    bytes_per_sec = out_rate * 2  # mono int16
    pre_bytes = int(bytes_per_sec * PREBUFFER_S)
    play_frames = max(32, int(out_rate * (PLAY_BLOCK_MS / 1000.0)))
    log_spk(f"Jitter prebuffer ≈ {int(PREBUFFER_S*1000)} ms ({pre_bytes} bytes). Play block: {PLAY_BLOCK_MS} ms ({play_frames} frames)")

    # Shared byte buffer guarded by a lock
    buf = bytearray()
    buf_lock = asyncio.Lock()

    async def feeder():
        # Pull 24 kHz chunks, convert to out_rate, append bytes
        acc = np.empty((0,), dtype=np.int16)
        while not stop_evt.is_set():
            try:
                chunk = await asyncio.wait_for(rx_audio_q.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            if chunk.size == 0:
                continue
            acc = np.concatenate((acc, chunk))
            # Convert in moderately sized lumps (~40–80 ms at model rate)
            min_samples = int(DOWNLINK_HZ * 0.04)
            if acc.size >= min_samples:
                out_i16 = resample_i16(acc, DOWNLINK_HZ, out_rate)
                async with buf_lock:
                    buf.extend(out_i16.tobytes())
                acc = np.empty((0,), dtype=np.int16)
        # flush any remainder
        if acc.size:
            out_i16 = resample_i16(acc, DOWNLINK_HZ, out_rate)
            async with buf_lock:
                buf.extend(out_i16.tobytes())

    feed_task = asyncio.create_task(feeder())

    # Wait for prebuffer
    t0 = time.time()
    while not stop_evt.is_set():
        async with buf_lock:
            bl = len(buf)
        if bl >= pre_bytes:
            break
        if (time.time() - t0) > 2.5:  # avoid long silence if nothing arrives
            log_spk("No audio within 2.5 s; starting anyway.")
            break
        await asyncio.sleep(0.01)

    if AUDIO_BACKEND == "sounddevice" and _sd is not None:
        def out_cb(outdata, frames, time_info, status):
            need = frames * 2
            give = b""
            # drain from jitter buffer
            # (We avoid long lock holds; slices are cheap on bytearray)
            # Use simple underrun padding if insufficient data
            # Lock must be synchronous; we are in a foreign callback thread; use try/except
            try:
                # Try non-async lock; we can't await here -> use simple best-effort
                pass
            except Exception:
                pass
            # Do a manual "critical section"
            if True:
                # WARNING: can't use asyncio.Lock here; use a simple heuristic without lock
                # Small risk of race, but bytearray ops are atomic enough for small reads.
                if len(buf) >= need:
                    give = bytes(buf[:need])
                    del buf[:need]
                else:
                    give = bytes(buf[:])
                    buf.clear()
            if len(give) < need:
                give += (np.zeros((need - len(give)) // 2, dtype=np.int16)).tobytes()
            outdata[:] = give

        with _sd.RawOutputStream(samplerate=out_rate, channels=1, dtype='int16',
                                 blocksize=play_frames, device=SPEAKER_DEVICE_INDEX,
                                 callback=out_cb):
            log_spk("Speaker started (sounddevice).")
            await stop_evt.wait()
    elif AUDIO_BACKEND == "soundcard" and _sc is not None:
        spk = _sc.default_speaker() if SPEAKER_DEVICE_INDEX is None else _sc.all_speakers()[SPEAKER_DEVICE_INDEX]
        log_spk("Speaker started (soundcard).")
        # soundcard player expects float32; we'll run a loop that sleeps per block
        with spk.player(samplerate=out_rate, channels=1) as p:
            frame_bytes = play_frames * 2
            while not stop_evt.is_set():
                # gather one block
                async with buf_lock:
                    if len(buf) >= frame_bytes:
                        b = bytes(buf[:frame_bytes]); del buf[:frame_bytes]
                    else:
                        b = bytes(buf); buf.clear()
                if len(b) < frame_bytes:
                    b += (np.zeros((frame_bytes - len(b)) // 2, dtype=np.int16)).tobytes()
                i16 = np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0
                p.play(i16.reshape(-1,1))
                await asyncio.sleep(PLAY_BLOCK_MS/1000.0)
    else:
        log_spk("No speaker backend available; audio will not play.")
        await stop_evt.wait()

    feed_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await feed_task
    log_spk("Speaker stopped.")

# ==============================
# Camera capture thread
# ==============================

class CameraThread(threading.Thread):
    daemon = True
    def __init__(self, loop: asyncio.AbstractEventLoop, gem: LiveAIOBridge, cam_index=0):
        super().__init__(name="CameraThread")
        self.loop = loop
        self.gem = gem
        self.cam_index = cam_index
        self.stop_evt = threading.Event()

    def stop(self):
        self.stop_evt.set()

    def run(self):
        if CAM_BACKEND != "opencv" or _cv2 is None:
            log_cam("OpenCV not available; camera preview disabled.")
            return
        cap = _cv2.VideoCapture(self.cam_index, _cv2.CAP_DSHOW) if os.name == "nt" else _cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            log_cam("Cannot open camera. Try CAM_INDEX=1/2/3.")
            return
        # Low-latency hints
        with contextlib.suppress(Exception):
            cap.set(_cv2.CAP_PROP_BUFFERSIZE, 1)
        with contextlib.suppress(Exception):
            cap.set(_cv2.CAP_PROP_FOURCC, _cv2.VideoWriter_fourcc(*"MJPG"))
        _cv2.namedWindow("Gemini Live (Q quits)", _cv2.WINDOW_NORMAL)
        _cv2.resizeWindow("Gemini Live (Q quits)", 960, 540)
        last_send = 0.0
        log_cam("Camera ready. Press Q to quit.")
        try:
            while not self.stop_evt.is_set():
                # Drain a few grabs to pick the freshest frame
                for _ in range(3):
                    cap.grab()
                ok, frame = cap.retrieve()
                if not ok:
                    time.sleep(0.02)
                    continue

                _cv2.imshow("Gemini Live (Q quits)", frame)
                key = _cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    # trigger global shutdown via loop
                    self.loop.call_soon_threadsafe(asyncio.get_event_loop().stop)
                    self.stop_evt.set()
                    break

                now = time.time()
                if (now - last_send) >= FRAME_INTERVAL_S:
                    last_send = now
                    # send jpeg via async client
                    fut = asyncio.run_coroutine_threadsafe(self.gem.send_jpeg_bgr(frame), self.loop)
                    # don't wait here; fire-and-forget
                time.sleep(0.001)
        finally:
            cap.release()
            _cv2.destroyAllWindows()
            log_cam("Camera closed.")

# ==============================
# Uplink sender (async)
# ==============================

async def uplink_sender(gem: LiveAIOBridge, mic_q: asyncio.Queue, stop_evt: asyncio.Event,
                        agc: RmsAgc):
    log_proc("Uplink sender started.")
    while not stop_evt.is_set():
        try:
            (raw_i16, src_rate) = await asyncio.wait_for(mic_q.get(), timeout=0.25)
        except asyncio.TimeoutError:
            continue
        # Noise gate then AGC, then resample to 16 kHz
        gated = noise_gate(raw_i16, NOISE_GATE_DBFS)
        processed = agc.process(gated, sample_rate=src_rate)
        pcm16k = resample_i16(processed, src_rate, UPLINK_HZ)
        await gem.send_pcm16(pcm16k)
    log_proc("Uplink sender stopped.")

# ==============================
# Main orchestration
# ==============================

async def run_main():
    # Friendly startup logs
    log_sys(f".env loaded: {DOTENV_OK}")
    log_sys(f"Libraries selected -> GenAI: {GENAI_BACKEND}, Audio: {AUDIO_BACKEND}, Camera: {CAM_BACKEND}, Resampler: {RESAMP_BACKEND}")

    api_key = os.getenv(ENV_API_KEY, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing {ENV_API_KEY}. Put it in your environment or a .env file.")

    if GENAI_BACKEND != "google-genai":
        raise RuntimeError("google-genai is required. If unavailable, a minimal httpx Live client is not implemented in this build.")

    # Open Gemini Live session (AUDIO modality, v1alpha)
    gem = LiveAIOBridge(api_key=api_key, model=GEMINI_MODEL)
    await gem.open()

    # Async constructs
    stop_evt = asyncio.Event()
    mic_async_q: asyncio.Queue = asyncio.Queue(maxsize=MIC_QUEUE_CAP)

    # Mic thread
    mic_thread = MicThread(asyncio.get_running_loop(), mic_async_q, device_index=MIC_DEVICE_INDEX)
    mic_thread.start()

    # Camera thread
    cam_thread = CameraThread(asyncio.get_running_loop(), gem, cam_index=CAM_INDEX)
    if CAM_BACKEND == "opencv":
        cam_thread.start()
    else:
        log_cam("Camera pipeline disabled (no backend).")

    # DSP bits
    agc = RmsAgc(AGC_TARGET_DBFS, AGC_MAX_GAIN_DB, AGC_ATTACK_MS, AGC_RELEASE_MS)

    # Async tasks
    send_task = asyncio.create_task(uplink_sender(gem, mic_async_q, stop_evt, agc))
    play_task = asyncio.create_task(speaker_player(gem.rx_audio_q, stop_evt))

    # Graceful shutdown signals
    def _sig_handler(*_):
        log_main("Signal received; shutting down...")
        asyncio.get_running_loop().call_soon_threadsafe(stop_evt.set)

    try:
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                asyncio.get_running_loop().add_signal_handler(s, _sig_handler)
            except NotImplementedError:
                pass  # Windows
    except Exception:
        pass

    # Run until the camera thread stops (on 'Q') or a fatal error occurs
    try:
        while cam_thread.is_alive():
            await asyncio.sleep(0.1)
        # If camera never started, wait until Ctrl+C
        if CAM_BACKEND != "opencv":
            await stop_evt.wait()
    finally:
        log_main("Shutting down...")
        stop_evt.set()
        mic_thread.stop()
        cam_thread.stop()
        # cancel async tasks
        for t in (send_task, play_task):
            t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(send_task, play_task)
        await gem.close()
        log_main("Clean exit.")

# Entrypoint
if __name__ == "__main__":
    try:
        asyncio.run(run_main())
    except Exception as e:
        logger.bind(tag="FATAL").exception(f"Unhandled error: {e}")
