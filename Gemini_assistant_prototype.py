# Gemini_assistant_prototype.py

"""
Gemini Live – Video call with Python (audio + video)

This script creates a "video call" with the Gemini 2.5 Flash Live API,
so the model can see your live camera, listen to your microphone, and respond with voice.
"""

import asyncio
import os
import sys
import signal
import time
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except Exception as e:
    sd = None
    print("[WARNING] sounddevice not available:", e, file=sys.stderr)

try:
    import cv2
except Exception as e:
    cv2 = None
    print("[WARNING] OpenCV (cv2) not available:", e, file=sys.stderr)

try:
    from google import genai
    from google.genai import types
except Exception as e:
    genai = None
    types = None
    print("[ERROR] google-genai missing. Install with: pip install -U google-genai", file=sys.stderr)

API_KEY_ENV = "GEMINI_API_KEY"
MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-live-2.5-flash-preview")

IN_RATE = 16000
OUT_RATE = 24000
CHANNELS = 1
IN_DTYPE = "int16"
OUT_DTYPE = "int16"

FRAME_W = int(os.environ.get("GLIVE_FRAME_W", 960))
FRAME_H = int(os.environ.get("GLIVE_FRAME_H", 720))
FRAME_INTERVAL = float(os.environ.get("GLIVE_FRAME_INTERVAL", 0.2))
JPEG_QUALITY = int(os.environ.get("GLIVE_JPEG_QUALITY", 70))

MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 3.0


class GracefulExit(SystemExit):
    """Controlled exit signal."""
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
    """Reads from the microphone and sends PCM16 16 kHz audio to the Live API."""
    if sd is None:
        print("[AUDIO] sounddevice not available; skipping capture.")
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
            # Discard to avoid blocking
            pass

    with sd.InputStream(samplerate=IN_RATE, channels=CHANNELS, dtype=IN_DTYPE,
                        blocksize=blocksize, callback=_in_cb):
        print("[AUDIO] Capturing microphone…")
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
                    print(f"[AUDIO] Error sending chunk: {e}", file=sys.stderr)
                    await asyncio.sleep(0.1)
        except GracefulExit:
            pass
        finally:
            try:
                await session.send_realtime_input(audio_stream_end=True)
            except Exception:
                pass
            print("[AUDIO] Capture stopped.")


async def audio_playback_task(session, stop_event: asyncio.Event) -> None:
    if sd is None:
        print("[AUDIO] sounddevice not available; audio will not be played.")
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

        # Fill 'buf' with chunks from the queue until the device's request is met
        while len(buf) < needed:
            try:
                chunk = out_q.get_nowait()
                buf.extend(chunk)
            except asyncio.QueueEmpty:
                break

        if len(buf) >= needed:
            # IMPORTANT! Copy to 'bytes' so NumPy does not reference 'buf'
            chunk = bytes(buf[:needed])
            del buf[:needed]
            arr = np.frombuffer(chunk, dtype=np.int16)
            outdata[:] = arr.reshape(-1, CHANNELS)
        else:
            # Underfill: play what is available and fill with zeros
            if len(buf) > 0:
                chunk = bytes(buf)
                buf.clear()
                part = np.frombuffer(chunk, dtype=np.int16)
                tmp = np.zeros(frames * CHANNELS, dtype=np.int16)
                n = min(tmp.size, part.size)
                tmp[:n] = part[:n]
                outdata[:] = tmp.reshape(-1, CHANNELS)
            else:
                outdata[:] = np.zeros((frames, CHANNELS), dtype=np.int16)

    with sd.OutputStream(
        samplerate=OUT_RATE, channels=CHANNELS, dtype=OUT_DTYPE,
        blocksize=OUT_RATE // 20,  # ~50 ms is fine; you can try //40 (~25 ms)
        callback=_out_fill
    ):
        print("[AUDIO] Playing responses…")

        # Transcription buffers
        in_text_parts, out_text_parts = [], []

        while not stop_event.is_set():
            try:
                async for message in session.receive():
                    # Model audio -> to queue (without losing leftovers)
                    if getattr(message, "data", None):
                        try:
                            out_q.put_nowait(message.data)
                        except asyncio.QueueFull:
                            # Avoid infinite latency: drop oldest chunk
                            try: _ = out_q.get_nowait()
                            except asyncio.QueueEmpty: pass
                            try: out_q.put_nowait(message.data)
                            except Exception: pass

                    sc = getattr(message, "server_content", None)

                    # Transcriptions: accumulate, DO NOT print fragments
                    if sc and getattr(sc, "input_transcription", None):
                        in_text_parts.append(sc.input_transcription.text)
                    if sc and getattr(sc, "output_transcription", None):
                        out_text_parts.append(sc.output_transcription.text)

                    # End of user activity -> dump input
                    if getattr(message, "activity_end", False) or getattr(message, "activityEnd", False):
                        if in_text_parts:
                            print("\n[YOU] ", "".join(in_text_parts))
                            in_text_parts.clear()

                    # Model's turn complete -> dump output (and clear)
                    if sc and getattr(sc, "turn_complete", False):
                        if out_text_parts:
                            print("\n[AI] ", "".join(out_text_parts))
                            out_text_parts.clear()

                await asyncio.sleep(0.05)
            except Exception as e:
                print(f"[AUDIO] Error receiving audio: {e}", file=sys.stderr)
                await asyncio.sleep(0.2)

        print("[AUDIO] Reception stopped.")


async def video_capture_task(session, stop_event: asyncio.Event) -> None:
    """Reads the webcam with OpenCV, encodes to JPEG, and sends frames as video."""
    if cv2 is None:
        print("[VIDEO] OpenCV not available; skipping video.")
        await stop_event.wait()
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[VIDEO] Could not open camera. Continuing without video.", file=sys.stderr)
        await stop_event.wait()
        return

    last_sent = 0.0
    print("[VIDEO] Sending frames… (Q to exit)")

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.05)
                continue

            # Display / Q key
            try:
                disp = cv2.resize(frame, (FRAME_W, FRAME_H))
                cv2.imshow("Gemini_assistant_prototype", disp)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    stop_event.set()
                    break
            except Exception:
                pass

            now = time.time()
            if now - last_sent < FRAME_INTERVAL:
                await asyncio.sleep(0.005)
                continue

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
                print(f"[VIDEO] Error sending frame: {e}", file=sys.stderr)
                await asyncio.sleep(0.01)
    except GracefulExit:
        pass
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[VIDEO] Capture stopped.")


async def run_session_once(client) -> None:
    """Creates a Live session and launches concurrent audio (in/out) and video tasks."""
    # Set media_resolution with enum when available
    if types is not None and hasattr(types, "MediaResolution"):
        media_res = types.MediaResolution.MEDIA_RESOLUTION_LOW
    else:
        media_res = "MEDIA_RESOLUTION_LOW"

    config = {
        "response_modalities": ["AUDIO"],
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": os.environ.get("GLIVE_VOICE", "Kore")}},
            "language_code": os.environ.get("GLIVE_LANG", "en-US"),
        },
        "media_resolution": media_res,
        "system_instruction": (
            "You are a voice assistant, concise and friendly. Describe in English what you see on the camera when asked. "
            "If you do not have enough visual context, ask to bring the object closer or improve the lighting."
        ),
        "input_audio_transcription": {},
        "output_audio_transcription": {},
    }

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        print(f"[LIVE] Session started with model: {MODEL}")
        stop_event = asyncio.Event()

        tasks = [
            asyncio.create_task(audio_capture_task(session, stop_event), name="audio_in"),
            asyncio.create_task(audio_playback_task(session, stop_event), name="audio_out"),
            asyncio.create_task(video_capture_task(session, stop_event), name="video"),
        ]

        # Initial greeting
        try:
            await session.send_client_content(
                turns={"role": "user", "parts": [{"text": "Hello, can you hear me?"}]},
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
            print("[LIVE] Session ended.")


async def main() -> None:
    if genai is None:
        print("[ERROR] Cannot continue without google-genai. Install with: pip install -U google-genai", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        print(f"[ERROR] Set the environment variable {API_KEY_ENV} with your API key.", file=sys.stderr)
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
                print(f"[ERROR] Session failed and maximum retries exceeded: {e}", file=sys.stderr)
                break
            wait = RETRY_BACKOFF_SEC * retries
            print(f"[WARN] Session error: {e}. Retrying in {wait:.1f}s…", file=sys.stderr)
            await asyncio.sleep(wait)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except GracefulExit:
        print("\n[EXIT] Shutdown signal received. Goodbye.")
    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted by user.")