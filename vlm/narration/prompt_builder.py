"""Builds LLM prompts from frame deltas and entity crops."""

from __future__ import annotations

import base64
import logging

import cv2
import numpy as np

from vlm.aggregation.delta_encoder import DeltaEncoder
from vlm.common.datatypes import FrameDelta

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """あなたは画面内容の分析者です。構造化された観測データを受け取り、画面上で何が起きているかを自然な日本語で説明してください。

ルール:
- エンティティはID (例: "E12") で参照してください。
- データに示されていることだけを説明し、推測や架空の行動を述べないでください。
- 簡潔に。重要な変化1つにつき1-2文。
- 変化がない場合は「変化なし」と報告してください。
- IDが再識別と注記されている場合、ナラティブの連続性に自然に組み込んでください。
- RELATIONS (空間関係) が提供された場合、エンティティ間の位置関係を自然に説明に含めてください。
- REL_CHANGES (+/-) が提供された場合、関係の変化を説明に反映してください。
- MEMORY (エピソード記憶) が提供された場合、過去のイベントとの連続性を考慮して説明してください。
- TIMELINE (時系列データ) が提供された場合、フレーム間の因果関係を推論し、一連の行動の流れを自然に説明してください。例えば「人がカップに近づき、それを持って歩いた」のような時間的因果関係を読み取ってください。
- 画面全体のスクリーンショットが提供された場合、画面に表示されているテキスト、UI要素、レイアウトを読み取って説明に含めてください。
- エンティティデータがない場合でも、スクリーンショットから画面の状態を説明してください。
- 1-2フレームだけ出現してすぐ消えるエンティティはノイズの可能性が高いので、特に言及しないでください。
- 検出データとスクリーンショットの内容が矛盾する場合は、スクリーンショットを信頼してください。
- アニメやイラストのキャラクターを実在の「人物」として説明しないでください。画面内コンテンツのキャラクターは「キャラクター」と呼んでください。
- 冷蔵庫、テディベア、信号機など、PC画面上に不自然な物体が検出された場合は、画面内のコンテンツ（動画、画像など）に映っている可能性を考慮してください。"""


class PromptBuilder:
    """Constructs prompts from FrameDelta + selected crops.

    Args:
        delta_encoder: DeltaEncoder for formatting text.
        jpeg_quality: JPEG compression quality for crop images.
    """

    def __init__(
        self,
        delta_encoder: DeltaEncoder,
        jpeg_quality: int = 70,
        screenshot_max_dim: int = 960,
        screenshot_jpeg_quality: int = 50,
    ):
        self._encoder = delta_encoder
        self._jpeg_quality = jpeg_quality
        self._screenshot_max_dim = screenshot_max_dim
        self._screenshot_jpeg_quality = screenshot_jpeg_quality

    def build(
        self,
        delta: FrameDelta | list[FrameDelta],
        context_text: str,
        key_crops: list[tuple[int, np.ndarray]] | None = None,
        relations_text: str = "",
        memory_text: str = "",
        screenshot: np.ndarray | None = None,
    ) -> list[dict]:
        """Build message list for LLM API call.

        Args:
            delta: Current frame delta, or list of accumulated deltas.
            context_text: Past narration context.
            key_crops: List of (track_id, crop_image) to include.
            relations_text: Scene graph relation text (compact or delta).
            memory_text: Working memory episodic text.
            screenshot: Full screen image to include for LLM vision.

        Returns:
            List of message dicts in OpenAI-compatible format.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add context from previous narrations
        if context_text:
            messages.append({
                "role": "user",
                "content": f"前回の観測:\n{context_text}",
            })

        # Build current observation
        if isinstance(delta, list):
            compact_text = self._encoder.to_temporal_text(delta)
        else:
            compact_text = self._encoder.to_compact_text(delta)

        # Append scene graph relations
        if relations_text:
            compact_text += f"\n{relations_text}"

        # Append working memory episodes
        if memory_text:
            compact_text += f"\n{memory_text}"

        content_parts: list[dict] = []

        # Add screenshot as the first image (before text)
        if screenshot is not None:
            b64 = self._encode_screenshot(screenshot)
            if b64:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                    },
                })
                content_parts.append({
                    "type": "text",
                    "text": "[画面全体のスクリーンショット]",
                })

        content_parts.append(
            {"type": "text", "text": f"現在の観測:\n{compact_text}"},
        )

        # Add key crops as images
        if key_crops:
            for track_id, crop in key_crops[:2]:  # max 2 crops
                b64 = self._encode_crop(crop)
                if b64:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                        },
                    })
                    content_parts.append({
                        "type": "text",
                        "text": f"[E{track_id}のクロップ画像]",
                    })

        messages.append({"role": "user", "content": content_parts})
        return messages

    def _encode_screenshot(self, image: np.ndarray) -> str | None:
        """Resize and encode full screenshot as base64 JPEG."""
        if image is None or image.size == 0:
            return None
        try:
            h, w = image.shape[:2]
            max_dim = self._screenshot_max_dim
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode(
                ".jpg", image,
                [cv2.IMWRITE_JPEG_QUALITY, self._screenshot_jpeg_quality],
            )
            return base64.b64encode(buf).decode("ascii")
        except Exception as e:
            logger.debug("Failed to encode screenshot: %s", e)
            return None

    def _encode_crop(self, crop: np.ndarray) -> str | None:
        """Encode crop as base64 JPEG."""
        if crop is None or crop.size == 0:
            return None
        try:
            _, buf = cv2.imencode(
                ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            )
            return base64.b64encode(buf).decode("ascii")
        except Exception as e:
            logger.debug("Failed to encode crop: %s", e)
            return None
