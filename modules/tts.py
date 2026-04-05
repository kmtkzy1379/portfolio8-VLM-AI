import aiohttp
import asyncio
from typing import Optional
from config import Config


class TTSHandler:
    def __init__(self):
        self.base_url = Config.VV_URL
        self.speaker = Config.VV_SPEAKER
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def generate_audio(self, text: str) -> bytes:
        if not text.strip():
            return None
        try:
            session = await self._get_session()
            params = {"text": text, "speaker": self.speaker}
            async with session.post(f"{self.base_url}/audio_query", params=params) as resp:
                if resp.status != 200: return None
                query_data = await resp.json()

            query_data['speedScale'] = Config.VV_SPEED
            query_data['pitchScale'] = Config.VV_PITCH

            async with session.post(
                f"{self.base_url}/synthesis",
                json=query_data,
                params={"speaker": self.speaker}
            ) as resp:
                if resp.status != 200: return None
                return await resp.read()
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
