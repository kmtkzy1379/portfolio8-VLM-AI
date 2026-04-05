"""
VisionAnalyzer - Groq Llama 4 Maverick による画像解析

base64 JPEG画像を受け取り、日本語で詳細な画面解析を返す。
Function Calling (get_screen_image) およびGameMode Oキー解析で使用。
"""
import asyncio
import base64
import logging
from typing import Optional

from groq import AsyncGroq, Groq

from config import Config

logger = logging.getLogger(__name__)

_VISION_PROMPT = """この画像の内容を日本語で簡潔に説明してください。以下を含めてください:
- 画面に表示されているテキスト
- キャラクター/人物の見た目
- UIの状態やレイアウト
- 重要な視覚的変化や注目点
具体的で簡潔に。"""


class VisionAnalyzer:
    """Groq Llama 4 Maverick によるマルチモーダル画像解析"""

    def __init__(self, model: Optional[str] = None):
        self._model = model or Config.VISION_ANALYSIS_MODEL
        self._async_client = AsyncGroq(api_key=Config.GROQ_API_KEY)
        self._sync_client = Groq(api_key=Config.GROQ_API_KEY)

    async def analyze(self, jpeg_bytes: bytes, prompt: Optional[str] = None) -> str:
        """非同期で画像解析を実行

        Args:
            jpeg_bytes: JPEG画像バイト
            prompt: カスタムプロンプト（省略時はデフォルト使用）

        Returns:
            解析結果テキスト。失敗時は空文字列
        """
        if not jpeg_bytes:
            return ""

        img_b64 = base64.b64encode(jpeg_bytes).decode('utf-8')

        try:
            response = await self._async_client.chat.completions.create(
                model=self._model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or _VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            }
                        }
                    ]
                }],
                max_tokens=500,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("VisionAnalyzer async error: %s", e)
            return ""

    def analyze_sync(self, jpeg_bytes: bytes, prompt: Optional[str] = None) -> str:
        """同期で画像解析を実行（VLMスレッドやFunction Callingから使用）

        Args:
            jpeg_bytes: JPEG画像バイト
            prompt: カスタムプロンプト

        Returns:
            解析結果テキスト。失敗時は空文字列
        """
        if not jpeg_bytes:
            return ""

        img_b64 = base64.b64encode(jpeg_bytes).decode('utf-8')

        try:
            response = self._sync_client.chat.completions.create(
                model=self._model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or _VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            }
                        }
                    ]
                }],
                max_tokens=500,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("VisionAnalyzer sync error: %s", e)
            return ""
