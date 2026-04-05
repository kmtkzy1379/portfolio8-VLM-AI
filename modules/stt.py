import io
import wave
import asyncio
from groq import Groq
from config import Config


class STTHandler:
    def __init__(self):
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model = "whisper-large-v3"


    async def transcribe(self, audio_bytes: bytes) -> str:
        if not audio_bytes:
            return ""


        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(Config.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(Config.SAMPLE_RATE)
            wf.writeframes(audio_bytes)
        wav_buffer.seek(0)
       
        loop = asyncio.get_running_loop()
        try:
            def _request():
                return self.client.audio.transcriptions.create(
                    file=("input.wav", wav_buffer.read()),
                    model=self.model,
                    language="ja",
                    response_format="text"
                )
            transcription = await loop.run_in_executor(None, _request)
            return transcription.strip()
        except Exception as e:
            print(f"STT Error: {e}")
            return ""



