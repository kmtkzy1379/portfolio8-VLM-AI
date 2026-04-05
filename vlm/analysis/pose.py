"""Pose estimation using MediaPipe Tasks API on pre-cropped images."""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from vlm.common.datatypes import SkeletonData

logger = logging.getLogger(__name__)

_MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

_MODEL_NAMES = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}


def _get_model_dir() -> Path:
    """Model directory next to the project root."""
    return Path(__file__).resolve().parents[3] / "models"


def _ensure_model(complexity: int) -> str:
    """Download pose model if not present. Returns path to .task file."""
    model_dir = _get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / _MODEL_NAMES[complexity]

    if model_path.exists():
        return str(model_path)

    url = _MODEL_URLS[complexity]
    logger.info("Downloading pose model: %s", _MODEL_NAMES[complexity])
    urllib.request.urlretrieve(url, str(model_path))
    logger.info("Pose model saved to: %s", model_path)
    return str(model_path)


class PoseEstimator:
    """Extracts skeleton keypoints from cropped entity images.

    Uses MediaPipe Tasks PoseLandmarker API in image mode.
    Crops are provided by the IDAuthority with stable track IDs.

    Args:
        model_complexity: 0=lite, 1=full, 2=heavy.
        min_detection_confidence: Minimum confidence for pose detection.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
    ):
        import mediapipe as mp

        model_path = _ensure_model(model_complexity)

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._mp = mp

    def estimate(self, crop: np.ndarray, track_id: int) -> Optional[SkeletonData]:
        """Run pose estimation on a pre-cropped image.

        Args:
            crop: BGR image cropped by IDAuthority.
            track_id: The stable track ID from IDAuthority.

        Returns:
            SkeletonData with keypoints, or None if no pose detected.
        """
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return None

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        h, w = crop.shape[:2]
        landmarks = result.pose_landmarks[0]
        keypoints = np.array(
            [
                [lm.x * w, lm.y * h, lm.visibility]
                for lm in landmarks
            ],
            dtype=np.float32,
        )

        pose_label = self._classify_pose(keypoints, h)

        return SkeletonData(
            track_id=track_id,
            keypoints=keypoints,
            pose_label=pose_label,
        )

    def close(self) -> None:
        self._landmarker.close()

    @staticmethod
    def _classify_pose(keypoints: np.ndarray, crop_height: int) -> str:
        """Simple pose classification from landmark positions.

        Uses relative positions of shoulders, hips, and knees.
        MediaPipe landmark indices:
          11=left_shoulder, 12=right_shoulder
          23=left_hip, 24=right_hip
          25=left_knee, 26=right_knee
        """
        try:
            shoulder_y = (keypoints[11, 1] + keypoints[12, 1]) / 2
            hip_y = (keypoints[23, 1] + keypoints[24, 1]) / 2
            knee_y = (keypoints[25, 1] + keypoints[26, 1]) / 2

            torso_len = abs(hip_y - shoulder_y)
            if torso_len < 5:
                return "unknown"

            torso_ratio = torso_len / crop_height

            if torso_ratio < 0.15:
                return "lying"

            knee_hip_dist = abs(knee_y - hip_y)
            if knee_hip_dist < torso_len * 0.5:
                return "sitting"

            return "standing"
        except (IndexError, ValueError):
            return "unknown"
