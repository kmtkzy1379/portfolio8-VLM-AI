import pyaudio
import asyncio
import io
import soundfile as sf
from config import Config


class AudioPlayer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.queue = asyncio.Queue()
        self.is_playing = False
        self.interrupt_signal = False
       
    async def play_worker(self):
        while True:
            try:
                wav_bytes = await self.queue.get()
                if self.interrupt_signal:
                    self.queue.task_done()
                    continue


                self.is_playing = True
                await self._play_audio(wav_bytes)
                self.is_playing = False
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Player Error: {e}")


    async def _play_audio(self, wav_bytes):
        with io.BytesIO(wav_bytes) as wav_io:
            data, samplerate = sf.read(wav_io, dtype='float32')


        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=samplerate,
            output=True
        )


        chunk_size = 1024
        loop = asyncio.get_running_loop()
       
        for i in range(0, len(data), chunk_size):
            if self.interrupt_signal:
                break
            chunk = data[i:i + chunk_size]
            await loop.run_in_executor(None, lambda: stream.write(chunk.tobytes()))


        stream.stop_stream()
        stream.close()


    def add_to_queue(self, audio_data: bytes):
        self.queue.put_nowait(audio_data)


    def interrupt(self):
        print(">> Player Interrupted!")
        self.interrupt_signal = True
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break
        asyncio.create_task(self._reset_interrupt())


    async def _reset_interrupt(self):
        await asyncio.sleep(0.2)
        self.interrupt_signal = False



