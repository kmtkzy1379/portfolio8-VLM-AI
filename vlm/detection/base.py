"""Abstract detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from vlm.common.datatypes import CapturedFrame, DetectionResult


class BaseDetector(ABC):
    """Base class for all object detectors."""

    @abstractmethod
    def detect(self, frame: CapturedFrame) -> DetectionResult:
        """Run detection on a frame and return results."""
        ...

    @property
    @abstractmethod
    def model_tier(self) -> str:
        """Return 'small' or 'mid'."""
        ...
