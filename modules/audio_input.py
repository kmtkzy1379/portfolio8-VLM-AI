import os
import sys
import pyaudio
import numpy as np
import torch
import asyncio
import time
import logging
from config import Config

logger = logging.getLogger(__name__)


def _load_silero_vad_direct():
    """Load Silero VAD JIT model directly, bypassing torch.hub (avoids torchaudio DLL issues)."""
    # Look for cached model from torch.hub
    hub_cache = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub",
                             "snakers4_silero-vad_master", "src", "silero_vad", "data")
    jit_path = os.path.join(hub_cache, "silero_vad.jit")

    if not os.path.isfile(jit_path):
        # Cache missing — download via torch.hub (first-time only)
        logger.info("Silero VAD cache not found, downloading via torch.hub...")
        torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=True, onnx=False)
        if not os.path.isfile(jit_path):
            raise FileNotFoundError(f"Silero VAD JIT model not found at {jit_path}")

    torch.set_num_threads(1)
    model = torch.jit.load(jit_path)
    model.eval()
    return model


class AudioInput:
    def __init__(self, callback_on_speech_start=None):
        self.chunk = 512
        self.format = pyaudio.paInt16
        self.channels = Config.CHANNELS
        self.rate = Config.SAMPLE_RATE

        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = False

        self.callback_on_speech_start = callback_on_speech_start
        self.speech_queue = asyncio.Queue()

        print("Loading Silero VAD model...")
        self.model = _load_silero_vad_direct()
        print("Silero VAD loaded.")

    def start(self):
        # Retry stream open up to 3 times (handles transient device-busy)
        last_err = None
        for attempt in range(3):
            try:
                self.stream = self.p.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    frames_per_buffer=self.chunk,
                )
                break
            except OSError as e:
                last_err = e
                logger.warning("PyAudio open attempt %d failed: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(0.5)
        else:
            raise OSError(f"Failed to open audio stream after 3 attempts: {last_err}") from last_err

        self.running = True
        asyncio.create_task(self._process_loop())


    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()


    async def get_audio(self):
        return await self.speech_queue.get()


    async def _process_loop(self):
        print("Microphone listening in background...")
        audio_buffer = []
        is_speaking = False
        silence_start_time = None
        loop = asyncio.get_running_loop()


        while self.running:
            try:
                # Non-blocking Read
                data = await loop.run_in_executor(
                    None,
                    lambda: self.stream.read(self.chunk, exception_on_overflow=False)
                )
            except Exception as e:
                print(f"Mic Error: {e}")
                await asyncio.sleep(0.1)
                continue


            audio_int16 = np.frombuffer(data, np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            tensor = torch.from_numpy(audio_float32)
           
            speech_prob = self.model(tensor, self.rate).item()


            if speech_prob > Config.VAD_THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                    print(">> Barge-in: User speaking")
                    if self.callback_on_speech_start:
                        if asyncio.iscoroutinefunction(self.callback_on_speech_start):
                            asyncio.create_task(self.callback_on_speech_start())
                        else:
                            self.callback_on_speech_start()
               
                silence_start_time = None
                audio_buffer.append(data)
            else:
                if is_speaking:
                    audio_buffer.append(data)
                    if silence_start_time is None:
                        silence_start_time = time.time()
                   
                    if time.time() - silence_start_time > Config.SILENCE_LIMIT:
                        is_speaking = False
                        full_audio = b''.join(audio_buffer)
                        audio_buffer = []
                       
                        if len(full_audio) > self.rate * 2 * 0.3:
                            self.speech_queue.put_nowait(full_audio)
                else:
                    pass
           
            await asyncio.sleep(0)



