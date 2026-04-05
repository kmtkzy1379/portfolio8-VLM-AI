"""GPU/CPU auto-detection and device management."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    CUDA = auto()       # NVIDIA GPU
    DIRECTML = auto()   # AMD GPU (Windows)
    CPU = auto()


@dataclass(frozen=True)
class DeviceInfo:
    device_type: DeviceType
    device_name: str
    torch_device: str          # "cuda", "cpu"
    onnx_providers: list[str]  # e.g. ["CUDAExecutionProvider", "CPUExecutionProvider"]


def detect_device(prefer: str = "auto") -> DeviceInfo:
    """Detect available compute device.

    Args:
        prefer: "auto", "cuda", "directml", or "cpu"
    """
    if prefer == "cuda" or (prefer == "auto" and _has_cuda()):
        name = _cuda_device_name()
        logger.info("Using NVIDIA CUDA: %s", name)
        return DeviceInfo(
            device_type=DeviceType.CUDA,
            device_name=name,
            torch_device="cuda",
            onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    if prefer == "directml" or (prefer == "auto" and _has_directml()):
        logger.info("Using AMD DirectML")
        return DeviceInfo(
            device_type=DeviceType.DIRECTML,
            device_name="DirectML",
            torch_device="cpu",  # PyTorch falls back to CPU; ONNX uses DirectML
            onnx_providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )

    logger.info("Using CPU")
    return DeviceInfo(
        device_type=DeviceType.CPU,
        device_name="CPU",
        torch_device="cpu",
        onnx_providers=["CPUExecutionProvider"],
    )


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _cuda_device_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CUDA (unknown)"


def _has_directml() -> bool:
    try:
        import onnxruntime as ort
        return "DmlExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False
