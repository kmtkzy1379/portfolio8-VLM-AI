"""Input validation utilities."""

from __future__ import annotations

import numpy as np

MAX_DIMENSION = 7680       # 8K
MAX_PIXELS = 33_177_600    # 8K total


def validate_frame(image: np.ndarray) -> None:
    """Validate a captured frame image. Raises ValueError on invalid input."""
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(image)}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 dtype, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 shape, got {image.shape}")
    h, w = image.shape[:2]
    if h > MAX_DIMENSION or w > MAX_DIMENSION:
        raise ValueError(f"Image too large: {w}x{h} (max {MAX_DIMENSION})")
    if h * w > MAX_PIXELS:
        raise ValueError(f"Too many pixels: {h * w} (max {MAX_PIXELS})")
    if h < 10 or w < 10:
        raise ValueError(f"Image too small: {w}x{h}")
