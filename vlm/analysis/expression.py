"""Facial expression detection on pre-cropped images."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from vlm.common.datatypes import ExpressionData

logger = logging.getLogger(__name__)


class ExpressionDetector:
    """Detects facial expressions from cropped entity images.

    Uses DeepFace for expression analysis. Crops are provided by
    the IDAuthority with stable track IDs.

    Args:
        backend: Face detector backend ("opencv", "retinaface", "mtcnn", "skip").
            "skip" means assume the crop IS a face (for pre-cropped face regions).
    """

    def __init__(self, backend: str = "opencv"):
        self._backend = backend
        self._deepface = None

    def _ensure_loaded(self) -> None:
        if self._deepface is None:
            from deepface import DeepFace
            self._deepface = DeepFace

    def analyze(self, crop: np.ndarray, track_id: int) -> Optional[ExpressionData]:
        """Analyze facial expression in a pre-cropped entity image.

        Args:
            crop: BGR image cropped by IDAuthority.
            track_id: The stable track ID from IDAuthority.

        Returns:
            ExpressionData, or None if no face/expression detected.
        """
        if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
            return None

        self._ensure_loaded()

        try:
            results = self._deepface.analyze(
                crop,
                actions=["emotion"],
                detector_backend=self._backend,
                enforce_detection=True,
                silent=True,
            )

            if not results:
                return None

            # DeepFace returns list; take first face
            result = results[0] if isinstance(results, list) else results
            emotion = result.get("emotion", {})

            if not emotion:
                return None

            dominant = result.get("dominant_emotion", "neutral")

            return ExpressionData(
                track_id=track_id,
                dominant_emotion=dominant,
                emotion_scores={k: round(v, 2) for k, v in emotion.items()},
            )

        except Exception as e:
            logger.debug("Expression detection failed for E%d: %s", track_id, e)
            return None
