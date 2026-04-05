"""YOLO detector with BoT-SORT tracking (CUDA/DirectML/CPU)."""

from __future__ import annotations

import logging
import os
import time

from ultralytics import YOLO

from vlm.common.datatypes import BoundingBox, CapturedFrame, DetectionResult
from vlm.common.device import DeviceInfo, DeviceType
from vlm.detection.base import BaseDetector

logger = logging.getLogger(__name__)


class YOLODetector(BaseDetector):
    """YOLO object detector with integrated BoT-SORT tracking.

    Uses ultralytics .track() to get detection + track IDs simultaneously,
    replacing the previous .predict() + external ByteTrack approach.

    Args:
        model_name: Model name (e.g. "yolo11m").
        device_info: Device configuration from detect_device().
        conf_threshold: Minimum confidence for detections.
        nms_threshold: NMS IoU threshold.
        input_size: Input image size for the model.
        tier: "small" or "mid" label for this detector.
        tracker_config: Path to tracker YAML config (e.g. "config/botsort_reid.yaml").
    """

    def __init__(
        self,
        model_name: str,
        device_info: DeviceInfo,
        conf_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        input_size: int = 640,
        tier: str = "small",
        class_whitelist: set[str] | None = None,
        min_box_area: float = 0,
        tracker_config: str = "config/botsort_reid.yaml",
    ):
        self._tier = tier
        self._conf = conf_threshold
        self._nms = nms_threshold
        self._imgsz = input_size
        self._whitelist = class_whitelist
        self._min_area = min_box_area
        self._tracker_config = tracker_config

        # Determine device string for ultralytics
        if device_info.device_type == DeviceType.CUDA:
            self._device = "cuda:0"
        else:
            self._device = "cpu"

        logger.info(
            "Loading %s model=%s device=%s tracker=%s",
            tier, model_name, self._device, tracker_config,
        )
        self._model = YOLO(f"{model_name}.pt")
        self._model.to(self._device)

        # Resolve tracker config path
        if not os.path.isabs(tracker_config):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            abs_tracker = os.path.join(project_root, tracker_config)
            if os.path.exists(abs_tracker):
                self._tracker_config = abs_tracker
                logger.info("Tracker config resolved: %s", abs_tracker)
            else:
                logger.warning(
                    "Tracker config NOT FOUND: %s (resolved: %s), falling back to botsort.yaml",
                    tracker_config, abs_tracker,
                )
                self._tracker_config = "botsort.yaml"
        else:
            logger.info("Tracker config (absolute): %s", tracker_config)

        # Warmup with .track() then reset tracker state
        self._warmup()

    def detect(self, frame: CapturedFrame) -> DetectionResult:
        t0 = time.perf_counter()
        results = self._model.track(
            frame.image,
            conf=self._conf,
            iou=self._nms,
            imgsz=self._imgsz,
            tracker=self._tracker_config,
            persist=True,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        boxes = []
        if results and results[0].boxes is not None:
            r_boxes = results[0].boxes
            # .track() returns .id with tracker IDs (None if no tracks yet)
            track_ids = r_boxes.id
            for i, box in enumerate(r_boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].item())
                cls_name = self._model.names.get(cls_id, f"class_{cls_id}")
                tid = -1
                if track_ids is not None:
                    val = track_ids[i]
                    if val is not None:
                        tid = int(val.item())
                boxes.append(
                    BoundingBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                        confidence=float(box.conf[0].item()),
                        class_id=cls_id,
                        class_name=cls_name,
                        track_id=tid,
                    )
                )

        # Apply class whitelist filter
        if self._whitelist is not None:
            boxes = [b for b in boxes if b.class_name in self._whitelist]
        # Apply minimum box area filter
        if self._min_area > 0:
            boxes = [b for b in boxes if b.area >= self._min_area]

        logger.debug(
            "frame=%d tier=%s detections=%d time=%.1fms",
            frame.metadata.frame_id,
            self._tier,
            len(boxes),
            elapsed_ms,
        )

        return DetectionResult(
            frame_id=frame.metadata.frame_id,
            boxes=boxes,
            model_tier=self._tier,
            inference_ms=elapsed_ms,
        )

    @property
    def model_tier(self) -> str:
        return self._tier

    def reset_tracker(self) -> None:
        """Reset tracker state (e.g. on scene cut)."""
        if hasattr(self._model, 'predictor') and self._model.predictor is not None:
            trackers = getattr(self._model.predictor, 'trackers', None)
            if trackers:
                for t in trackers:
                    t.reset()
                logger.info("Tracker state reset for %s", self._tier)

    def _warmup(self) -> None:
        """Warmup with .track() to initialize tracker, then reset state."""
        import numpy as np

        dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        self._model.track(
            dummy, conf=0.5, persist=True,
            tracker=self._tracker_config, verbose=False,
        )
        self.reset_tracker()
        logger.info("%s model warmup complete (tracker reset)", self._tier)
