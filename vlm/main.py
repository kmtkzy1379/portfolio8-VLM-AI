"""Main pipeline orchestrator - wires all modules together.

Brain-inspired processing pipeline:
  V1 Saliency → V2 Predictive Coding → Detection → Tracking
  → MT/V5 Optical Flow → Per-ID Analysis → Scene Graph
  → Working Memory → Delta Encoding → LLM Narration
"""

from __future__ import annotations

import logging
import queue
import signal
import sys
import threading
import time
from typing import Optional

from vlm.aggregation.delta_encoder import DeltaEncoder
from vlm.aggregation.feature_store import FeatureStore
from vlm.aggregation.scene_graph import SceneGraphBuilder
from vlm.aggregation.token_budget import TokenBudgetManager
from vlm.analysis.expression import ExpressionDetector
from vlm.analysis.optical_flow import OpticalFlowMotion
from vlm.analysis.per_id_analyzer import PerIDAnalyzer
from vlm.analysis.pose import PoseEstimator
from vlm.capture.change_detector import ChangeDetector
from vlm.capture.predictive_coder import PredictiveCoder
from vlm.capture.saliency import SaliencyDetector
from vlm.capture.screen import ScreenCapture
from vlm.common.config import get_nested, load_config
from vlm.common.datatypes import (
    ChangeLevel,
    EntityFeatures,
    FrameDelta,
    NarrationRequest,
    NarrationResult,
)
from vlm.common.device import detect_device
from vlm.detection.yolo_detector import YOLODetector
from vlm.narration.llm_client import NarrationEngine
from vlm.tracking.id_authority import IDAuthority
from vlm.tracking.track_store import TrackStore
from vlm.tracking.working_memory import WorkingMemory

logger = logging.getLogger(__name__)

_SENTINEL = object()  # poison pill for thread shutdown


class Pipeline:
    """Main recognition pipeline with brain-inspired processing.

    Processing stages (mapped to visual cortex areas):
      1. V1: Saliency detection (bottom-up attention)
      2. V2: Predictive coding (change region detection)
      3. IT: Object detection (YOLOv8)
      4. Central: Tracking (ByteTrack ID authority)
      5. MT/V5: Optical flow (pixel-level motion)
      6. IT+: Per-ID analysis (pose, expression)
      7. Parietal: Scene graph (spatial relations)
      8. Hippocampus: Working memory (Re-ID, episodes)
      9. Prefrontal: Delta encoding + LLM narration
    """

    def __init__(self, config_path: Optional[str] = None):
        self._config = load_config(config_path)
        self._running = False

        # Detect device
        device_pref = get_nested(self._config, "device.prefer", "auto")
        self._device = detect_device(device_pref)
        logger.info("Device: %s (%s)", self._device.device_name, self._device.device_type.name)

        # Initialize all modules
        self._init_capture()
        self._init_detection()
        self._init_tracking()
        self._init_analysis()
        self._init_aggregation()
        self._init_narration()

        # Scene cut detection config
        self._scene_cut_count = get_nested(self._config, "change_detection.scene_cut_count", 5)
        self._scene_cut_window = get_nested(self._config, "change_detection.scene_cut_window", 5.0)

        # Latest frame image (for screenshot capture by VLMBridge)
        self._latest_frame_image = None

        # State
        self._rapid_change_count = 0
        self._rapid_change_window_start = 0.0
        self._frames_since_narration = 0
        self._accumulated_deltas: list[FrameDelta] = []

        # Phase 3.1: structured counts accumulation (Hierarchical surprise の入力用)
        self._accum_rel_added: int = 0
        self._accum_rel_removed: int = 0
        self._accum_episodes_baseline: int = 0  # 前回 narration 時点の episode 数

        # Threading state
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._narration_queue: queue.Queue = queue.Queue(maxsize=4)
        self._narration_result_queue: queue.Queue = queue.Queue(maxsize=8)
        self._capture_thread: Optional[threading.Thread] = None
        self._llm_thread: Optional[threading.Thread] = None

    # ── Initialization ──

    def _init_capture(self) -> None:
        self._capture = ScreenCapture(
            monitor=get_nested(self._config, "capture.monitor", 1),
            target_fps=get_nested(self._config, "capture.target_fps", 2.0),
            max_dimension=get_nested(self._config, "capture.max_dimension", 1920),
        )
        self._change_detector = ChangeDetector(
            phash_threshold=get_nested(self._config, "change_detection.phash_threshold", 12),
            ssim_threshold_major=get_nested(self._config, "change_detection.ssim_threshold_major", 0.50),
            ssim_threshold_moderate=get_nested(self._config, "change_detection.ssim_threshold_moderate", 0.80),
            ssim_downscale=get_nested(self._config, "change_detection.ssim_downscale", 256),
            periodic_interval=get_nested(self._config, "change_detection.periodic_interval", 10),
        )
        # Brain V2: Predictive coding - detect WHERE changes occurred
        self._predictive_coder = PredictiveCoder(
            diff_threshold=get_nested(self._config, "predictive_coding.diff_threshold", 30),
            min_region_area=get_nested(self._config, "predictive_coding.min_region_area", 500),
            merge_distance=get_nested(self._config, "predictive_coding.merge_distance", 50),
        )
        # Brain V1: Saliency - detect WHAT is visually prominent
        self._saliency = SaliencyDetector(
            saliency_threshold=get_nested(self._config, "saliency.min_score", 0.4),
            min_region_area=get_nested(self._config, "saliency.min_region_area", 500),
            change_weight=get_nested(self._config, "saliency.change_weight", 0.6),
            saliency_weight=get_nested(self._config, "saliency.spatial_weight", 0.4),
        )

    def _init_detection(self) -> None:
        small_model = get_nested(self._config, "detection.small_model", "yolov8n")
        mid_model = get_nested(self._config, "detection.mid_model", "yolov8m")
        conf = get_nested(self._config, "detection.confidence_threshold", 0.35)
        nms = get_nested(self._config, "detection.nms_threshold", 0.45)
        imgsz = get_nested(self._config, "detection.input_size", 640)
        min_box_area = get_nested(self._config, "detection.min_box_area", 0)
        whitelist_raw = get_nested(self._config, "detection.class_whitelist", None)
        class_whitelist = set(whitelist_raw) if whitelist_raw else None

        tracker_config = get_nested(self._config, "tracking.tracker_config", "config/botsort_reid.yaml")

        logger.info("Loading small detector: %s", small_model)
        self._small_detector = YOLODetector(
            small_model, self._device, conf, nms, imgsz, tier="small",
            class_whitelist=class_whitelist, min_box_area=min_box_area,
            tracker_config=tracker_config,
        )
        self._mid_detector: Optional[YOLODetector] = None
        self._small_model_name = small_model
        self._mid_model_name = mid_model
        self._det_conf = conf
        self._det_nms = nms
        self._det_imgsz = imgsz
        self._det_class_whitelist = class_whitelist
        self._det_min_box_area = min_box_area
        self._det_tracker_config = tracker_config

    def _init_tracking(self) -> None:
        self._id_authority = IDAuthority(
            min_hits=get_nested(self._config, "tracking.min_hits", 5),
            max_entities=get_nested(self._config, "tracking.max_entities", 100),
        )
        self._track_store = TrackStore()
        # Brain Hippocampus: Working memory for Re-ID + episodic memory
        self._working_memory = WorkingMemory(
            memory_duration=get_nested(self._config, "working_memory.memory_duration", 30.0),
            max_remembered=get_nested(self._config, "working_memory.max_remembered", 50),
            reid_threshold=get_nested(self._config, "working_memory.reid_threshold", 0.6),
            max_episodes=get_nested(self._config, "working_memory.max_episodes", 100),
        )

    def _init_analysis(self) -> None:
        pose = PoseEstimator(
            model_complexity=get_nested(self._config, "analysis.pose_model_complexity", 1),
            min_detection_confidence=get_nested(self._config, "analysis.pose_min_confidence", 0.5),
        )
        expr = ExpressionDetector()
        self._analyzer = PerIDAnalyzer(
            pose_estimator=pose,
            expression_detector=expr,
            skip_iou_threshold=get_nested(self._config, "analysis.skip_if_iou_above", 0.9),
            skip_min_frames=get_nested(self._config, "analysis.skip_if_frames_below", 5),
            expression_enabled=get_nested(self._config, "analysis.expression_enabled", True),
        )
        # Brain MT/V5: Optical flow for pixel-level motion
        self._optical_flow = OpticalFlowMotion()

    def _init_aggregation(self) -> None:
        self._feature_store = FeatureStore()
        self._delta_encoder = DeltaEncoder(
            feature_store=self._feature_store,
            coordinate_precision=get_nested(self._config, "aggregation.coordinate_precision", 5),
            min_movement=get_nested(self._config, "aggregation.min_movement_threshold", 10.0),
            min_lifetime=get_nested(self._config, "aggregation.min_entity_lifetime", 0),
        )
        # Brain Parietal: Scene graph for spatial relations
        self._scene_graph = SceneGraphBuilder(
            near_threshold=get_nested(self._config, "scene_graph.near_threshold", 200.0),
            overlap_iou_threshold=get_nested(self._config, "scene_graph.overlap_iou_threshold", 0.15),
            containment_threshold=get_nested(self._config, "scene_graph.containment_threshold", 0.7),
        )
        self._token_budget = TokenBudgetManager(
            max_tokens=get_nested(self._config, "aggregation.max_tokens", 4000),
        )
        self._batch_size = get_nested(self._config, "aggregation.batch_frames", 5)

    def _init_narration(self) -> None:
        self._send_screenshot = get_nested(self._config, "narration.send_screenshot", True)
        self._narration = NarrationEngine(
            model=get_nested(self._config, "narration.model", "gemini/gemini-2.0-flash"),
            fallback_model=get_nested(self._config, "narration.fallback_model", None),
            delta_encoder=self._delta_encoder,
            min_interval=get_nested(self._config, "narration.min_interval_seconds", 5.0),
            max_context_entries=get_nested(self._config, "narration.max_context_entries", 3),
            max_crops=get_nested(self._config, "narration.max_crops", 2),
            jpeg_quality=get_nested(self._config, "narration.crop_jpeg_quality", 70),
            screenshot_max_dim=get_nested(self._config, "narration.screenshot_max_dimension", 960),
            screenshot_jpeg_quality=get_nested(self._config, "narration.screenshot_jpeg_quality", 50),
        )

    # ── Thread loops ──

    def _capture_loop(self) -> None:
        """Thread 1: Capture frames and feed into frame_queue."""
        try:
            for frame in self._capture.stream_until(self._stop_event):
                if self._stop_event.is_set():
                    break
                try:
                    self._frame_queue.put(frame, timeout=0.5)
                except queue.Full:
                    # Drop stale frame, keep freshest data
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass
        except Exception:
            logger.exception("Capture thread error")
        finally:
            # Signal processing thread that capture is done
            try:
                self._frame_queue.put(_SENTINEL, timeout=1)
            except queue.Full:
                pass
            logger.debug("Capture thread exiting")

    def _llm_loop(self) -> None:
        """Thread 3: Consume narration requests and call LLM."""
        try:
            while True:
                try:
                    request = self._narration_queue.get(timeout=1.0)
                except queue.Empty:
                    if self._stop_event.is_set():
                        break
                    continue

                if request is _SENTINEL:
                    break

                if request.is_reset:
                    self._narration.clear_context()
                    continue

                narration = self._narration.narrate(
                    request.deltas, request.key_crops,
                    relations_text=request.relations_text,
                    memory_text=request.memory_text,
                    screenshot=request.screenshot,
                    is_scene_change=getattr(request, "is_scene_change", False),
                )
                if narration:
                    # Determine the highest change level across all deltas
                    max_change = ChangeLevel.NONE
                    for d in request.deltas:
                        if d.change_level.value > max_change.value:
                            max_change = d.change_level

                    # Phase 2: Build versioned envelope of structured metadata
                    # for downstream FEP-inspired feedback (Eve side).
                    # change_magnitude_* are visual surprise proxies, not true
                    # neuroscientific prediction errors.
                    if request.deltas:
                        avg_mags = [d.change_magnitude_avg for d in request.deltas]
                        max_mags = [d.change_magnitude_max for d in request.deltas]
                        n_regs_list = [d.n_regions for d in request.deltas]
                        ratios = [d.change_area_ratio for d in request.deltas]
                        period_avg = sum(avg_mags) / len(avg_mags)
                        period_max = max(max_mags)
                        period_n_regs_max = max(n_regs_list)
                        period_area_ratio_max = max(ratios)
                    else:
                        period_avg = period_max = 0.0
                        period_n_regs_max = 0
                        period_area_ratio_max = 0.0

                    # Phase 3.1: Saliency-weighted aggregation
                    # NOTE: change_score > 0 の region のみで重み付き平均を取る。
                    # salient-only region (change_score=0) を含めると静的 UI の
                    # saliency が分母に効いて avg が不自然に下がるため除外。
                    all_scored: list = []
                    for d in request.deltas:
                        if d.scored_regions:
                            all_scored.extend(d.scored_regions)
                    movers = [sr for sr in all_scored if sr.change_score > 0.0]
                    if movers:
                        total_w = sum(sr.w * sr.h * sr.saliency_score for sr in movers)
                        if total_w > 0:
                            sw_avg = sum(
                                (sr.change_score * 255.0) * sr.w * sr.h * sr.saliency_score
                                for sr in movers
                            ) / total_w
                        else:
                            sw_avg = 0.0
                        sw_max = max(sr.change_score * 255.0 for sr in movers)
                    else:
                        sw_avg = sw_max = 0.0

                    # Top Salient Regions は全 region (静的含む) で combined_score 降順
                    top_5 = sorted(
                        all_scored, key=lambda sr: sr.combined_score, reverse=True
                    )[:5]
                    top_salient = [
                        {
                            "bbox": [sr.x, sr.y, sr.w, sr.h],
                            "saliency": round(sr.saliency_score, 3),
                            "change": round(sr.change_score, 3),
                            "combined": round(sr.combined_score, 3),
                            "is_static": sr.change_score == 0.0,
                        }
                        for sr in top_5
                    ]

                    # Phase 3.1: Structured counts (Phase 3.2 hierarchical surprise の入力用)
                    n_entity_new = sum(
                        1 for d in request.deltas for ed in d.entity_deltas if ed.is_new
                    )
                    n_entity_lost = sum(
                        1 for d in request.deltas for ed in d.entity_deltas if ed.is_lost
                    )
                    n_entity_changed = sum(
                        1 for d in request.deltas for ed in d.entity_deltas
                        if not ed.is_new and not ed.is_lost
                    )
                    # Episode count: 前回 narration 以降の追加分
                    current_episode_total = len(self._working_memory._episodes)
                    n_new_episodes = max(
                        0, current_episode_total - self._accum_episodes_baseline
                    )

                    meta = {
                        "version": "v2",
                        # Phase 2 v1 fields (互換)
                        "change_magnitude_avg": period_avg,
                        "change_magnitude_max": period_max,
                        "n_regions_max": period_n_regs_max,
                        "change_area_ratio_max": period_area_ratio_max,
                        "relations_text": request.relations_text or "",
                        "memory_text": request.memory_text or "",
                        "entity_delta_text": self._delta_encoder.to_temporal_text(
                            request.deltas
                        ),
                        "n_deltas": len(request.deltas),
                        "frame_id": request.frame_id,
                        # Phase 3.1 v2: saliency precision weighting
                        "saliency_weighted_avg": sw_avg,
                        "saliency_weighted_max": sw_max,
                        "top_salient_regions": top_salient,
                        "n_movers": len(movers),
                        # Phase 3.1 v2: structured counts (hierarchical surprise input)
                        "change_level_name": max_change.name,
                        "n_entity_new": n_entity_new,
                        "n_entity_lost": n_entity_lost,
                        "n_entity_changed": n_entity_changed,
                        "n_relation_added": self._accum_rel_added,
                        "n_relation_removed": self._accum_rel_removed,
                        "n_memory_episodes": n_new_episodes,
                    }

                    result = NarrationResult(
                        narration=narration,
                        change_level=max_change,
                        meta=meta,
                        timestamp_ms=request.deltas[-1].timestamp_ms if request.deltas else 0.0,
                        frame_id=request.frame_id,
                    )
                    try:
                        self._narration_result_queue.put_nowait(result)
                    except queue.Full:
                        # Drop oldest result to make room
                        try:
                            self._narration_result_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self._narration_result_queue.put_nowait(result)
                        except queue.Full:
                            pass

                    # Phase 3.1: 累積カウンタをリセット
                    self._accum_rel_added = 0
                    self._accum_rel_removed = 0
                    self._accum_episodes_baseline = current_episode_total
        except Exception:
            logger.exception("LLM thread error")
        finally:
            logger.debug("LLM thread exiting")

    # ── Main loop ──

    def run(self) -> None:
        """Run the 3-thread brain-inspired pipeline loop."""
        self._running = True
        self._stop_event.clear()
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._handle_signal)

        # Start background threads
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="capture", daemon=True,
        )
        self._llm_thread = threading.Thread(
            target=self._llm_loop, name="llm", daemon=True,
        )
        self._capture_thread.start()
        self._llm_thread.start()

        logger.info("Pipeline started (3-thread mode). Press Ctrl+C to stop.")

        try:
            while self._running:
                # Drain any completed narration results
                self._drain_narration_results()

                # Pull next frame from capture thread
                try:
                    frame = self._frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if frame is _SENTINEL:
                    break

                # Store latest frame image for VLMBridge screenshot capture
                self._latest_frame_image = frame.image

                t0 = time.perf_counter()

                # ── Stage 1: V1+V2 - Change detection + Predictive coding ──
                change_level = self._change_detector.evaluate(frame)

                if change_level == ChangeLevel.NONE:
                    continue

                # Rapid scene cut detection
                if change_level == ChangeLevel.MAJOR:
                    if self._detect_rapid_changes():
                        self._handle_scene_cut()
                        continue

                # V2: Compute change regions (WHERE changed)
                change_regions = self._predictive_coder.compute_change_regions(frame.image)

                # Phase 2: Aggregate visual surprise proxy from change regions.
                # NOTE: change_magnitude is a pixel-difference proxy, NOT a true
                # neuroscientific prediction error. Use area-weighted mean for
                # robustness against camera shake / lighting variations.
                if change_regions:
                    total_area = sum(r.area for r in change_regions)
                    if total_area > 0:
                        cm_avg = (
                            sum(r.change_magnitude * r.area for r in change_regions)
                            / total_area
                        )
                    else:
                        cm_avg = 0.0
                    cm_max = max(r.change_magnitude for r in change_regions)
                    n_regs = len(change_regions)
                    h, w = frame.image.shape[:2]
                    total_pixels = max(h * w, 1)
                    area_ratio = total_area / total_pixels
                else:
                    cm_avg = cm_max = 0.0
                    n_regs = 0
                    area_ratio = 0.0

                # V1: Combine with saliency (WHAT is prominent)
                scored_regions = self._saliency.combine_with_changes(
                    frame.image, change_regions
                )

                # Log attention info
                if scored_regions:
                    top = scored_regions[0]
                    logger.debug(
                        "Top attention: (%d,%d,%d,%d) sal=%.2f chg=%.2f comb=%.2f",
                        top.x, top.y, top.w, top.h,
                        top.saliency_score, top.change_score, top.combined_score,
                    )

                # ── Stage 2: IT - Object detection ──
                if change_level == ChangeLevel.MAJOR:
                    self._lazy_load_mid_detector()
                    detector = self._mid_detector or self._small_detector
                else:
                    detector = self._small_detector
                detections = detector.detect(frame)

                # ── Stage 3: Central tracking (ID Authority) ──
                tracking_state = self._id_authority.update(frame, detections)

                # ── Stage 4: MT/V5 - Optical flow (full-frame) ──
                # Skip expensive Farneback computation for MINOR changes;
                # update reference frame only to maintain continuity.
                if change_level == ChangeLevel.MINOR:
                    self._optical_flow.update_reference_only(frame.image)
                else:
                    self._optical_flow.update_frame(frame.image)

                # ── Stage 5: Hippocampus - Working memory (Re-ID + episodes) ──
                # Process lost entities
                for eid in tracking_state.lost_ids:
                    entity = tracking_state.entities.get(eid)
                    if entity:
                        self._working_memory.on_entity_lost(entity, frame.metadata.frame_id)

                # Process new entities (attempt Re-ID)
                reid_notes: dict[int, str] = {}
                for eid in tracking_state.new_ids:
                    entity = tracking_state.entities.get(eid)
                    if entity:
                        match = self._working_memory.on_entity_new(
                            entity, frame.metadata.frame_id
                        )
                        if match:
                            reid_notes[eid] = (
                                f"E{eid} is likely former E{match.old_track_id} "
                                f"(sim={match.similarity})"
                            )

                # ── Stage 6: IT+ - Per-ID analysis (pose, expression, motion) ──
                features: dict[int, EntityFeatures] = {}
                for eid, entity in tracking_state.entities.items():
                    if not entity.is_active:
                        continue
                    prev = self._feature_store.get_latest(eid)
                    feat = self._analyzer.analyze(entity, frame.metadata.frame_id, prev)

                    # Override motion with optical flow data if available
                    if self._optical_flow.has_flow:
                        flow_motion = self._optical_flow.compute_entity_motion(
                            eid, entity.bbox
                        )
                        feat.motion = flow_motion

                    features[eid] = feat
                    self._feature_store.store(feat)
                    self._track_store.store(entity)

                # ── Stage 7: Parietal - Scene graph (spatial relations) ──
                rel_added, rel_removed = self._scene_graph.build_delta(
                    tracking_state.entities
                )

                # ── Stage 8: Prefrontal - Delta encoding ──
                scene_label = self._heuristic_scene_label(
                    features, change_level, area_ratio
                )
                delta = self._delta_encoder.encode(
                    tracking_state, features, change_level,
                    scene_label=scene_label,
                    timestamp_ms=frame.metadata.timestamp_ms,
                    change_magnitude_avg=cm_avg,
                    change_magnitude_max=cm_max,
                    n_regions=n_regs,
                    change_area_ratio=area_ratio,
                )

                # Phase 3.1: ScoredRegion を delta に格納 (Saliency precision weighting source)
                delta.scored_regions = list(scored_regions)

                # Phase 3.1: 期間内の関係増減を累積 (NarrationRequest 送信時にリセット)
                self._accum_rel_added += len(rel_added)
                self._accum_rel_removed += len(rel_removed)

                elapsed_ms = (time.perf_counter() - t0) * 1000

                # ── Output ──
                active = len([e for e in tracking_state.entities.values() if e.is_active])
                logger.info(
                    "frame=%d change=%s active=%d new=%d lost=%d regions=%d rels=+%d/-%d time=%.0fms",
                    frame.metadata.frame_id,
                    change_level.name,
                    active,
                    len(tracking_state.new_ids),
                    len(tracking_state.lost_ids),
                    len(scored_regions),
                    len(rel_added),
                    len(rel_removed),
                    elapsed_ms,
                )

                # Print compact output
                if delta.entity_deltas:
                    compact = self._delta_encoder.to_compact_text(delta)
                    print(compact)

                    # Print Re-ID notes
                    for note in reid_notes.values():
                        print(f"  REID: {note}")

                    # Print spatial relation changes
                    rel_text = self._scene_graph.to_delta_text(rel_added, rel_removed)
                    if rel_text:
                        print(rel_text)

                    print()

                # ── Stage 9: LLM Narration (non-blocking submit) ──
                self._accumulated_deltas.append(delta)
                self._frames_since_narration += 1

                if self._should_narrate(change_level):
                    key_crops = self._select_key_crops(tracking_state)
                    all_rels = self._scene_graph.build(tracking_state.entities)
                    rel_text_for_llm = self._scene_graph.to_compact_text(all_rels)
                    mem_text_for_llm = self._working_memory.get_episodes_text(n=5)
                    request = NarrationRequest(
                        deltas=list(self._accumulated_deltas),
                        key_crops=key_crops,
                        relations_text=rel_text_for_llm,
                        memory_text=mem_text_for_llm,
                        screenshot=frame.image.copy() if self._send_screenshot else None,
                        frame_id=frame.metadata.frame_id,
                        # Bug-C1: MAJOR 変化時は蓄積 delta が「切替前」の記録に偏るため、
                        # ナレーション側に screenshot 優先を指示する（古い画面描写の防止）。
                        is_scene_change=(change_level == ChangeLevel.MAJOR),
                    )
                    try:
                        self._narration_queue.put_nowait(request)
                        self._frames_since_narration = 0
                        self._accumulated_deltas.clear()
                    except queue.Full:
                        logger.debug("Narration queue full, skipping LLM request")

        finally:
            self._shutdown()

    # ── Helpers ──

    def _heuristic_scene_label(
        self,
        features: dict,
        change_level: ChangeLevel,
        area_ratio: float,
    ) -> Optional[str]:
        """軽量ヒューリスティックで scene_label を決める。

        - 検出 entity なし: "static" (変化が小さい) / "transitioning" (大変化)
        - person が支配的: "person-focused"
        - 複数クラスが混在: "multi-entity"
        - 単一非 person クラスが支配的: "{class}-dominant"

        外部モデル無し。あくまで参考シグナルで、narration LLM 側で過信しない指示済み。
        """
        try:
            if not features:
                if change_level == ChangeLevel.MAJOR or area_ratio > 0.3:
                    return "transitioning"
                return "static"

            class_counts: dict[str, int] = {}
            for feat in features.values():
                cls = feat.attributes.get("class", "unknown") if hasattr(feat, "attributes") else "unknown"
                class_counts[cls] = class_counts.get(cls, 0) + 1

            if not class_counts:
                return "static"

            total = sum(class_counts.values())
            top_class, top_count = max(class_counts.items(), key=lambda kv: kv[1])
            top_share = top_count / total if total > 0 else 0.0

            if top_class == "person" and top_share >= 0.5:
                return "person-focused"
            if top_share >= 0.7:
                return f"{top_class}-dominant"
            return "multi-entity"
        except Exception:
            return None

    def _should_narrate(self, change_level: ChangeLevel) -> bool:
        has_entity_deltas = any(
            d.entity_deltas for d in self._accumulated_deltas
        )
        if change_level == ChangeLevel.MAJOR:
            return has_entity_deltas or self._send_screenshot
        if self._frames_since_narration >= self._batch_size:
            return has_entity_deltas or self._send_screenshot
        return False

    def _detect_rapid_changes(self) -> bool:
        now = time.monotonic()
        if now - self._rapid_change_window_start > self._scene_cut_window:
            self._rapid_change_count = 0
            self._rapid_change_window_start = now
        self._rapid_change_count += 1
        return self._rapid_change_count >= self._scene_cut_count

    def _drain_narration_results(self) -> None:
        """Non-blocking drain of completed narration results.

        Phase 3: result は NarrationResult dataclass、dict envelope、
        legacy tuple (Phase 1/2) のいずれも受け付ける。VLM 単体起動時
        (このメソッドが直接 narration を表示する場合) のクラッシュ回避。
        """
        while True:
            try:
                result = self._narration_result_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(result, NarrationResult):
                narration = result.narration
            elif isinstance(result, dict):
                narration = result.get("narration", "")
            elif isinstance(result, tuple):
                if len(result) >= 1:
                    narration = result[0]
                else:
                    narration = ""
            else:
                narration = str(result)
            print(f"\n{'='*60}")
            print("[LLM Narration]")
            print(narration)
            print(f"{'='*60}\n")

    def _shutdown(self) -> None:
        """Gracefully shut down all threads."""
        logger.info("Shutting down pipeline threads...")
        self._stop_event.set()

        # Signal LLM thread to exit
        try:
            self._narration_queue.put_nowait(_SENTINEL)
        except queue.Full:
            try:
                self._narration_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._narration_queue.put_nowait(_SENTINEL)
            except queue.Full:
                pass

        # Join threads
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=3.0)
            if self._capture_thread.is_alive():
                logger.warning("Capture thread did not exit in time")

        if self._llm_thread and self._llm_thread.is_alive():
            self._llm_thread.join(timeout=5.0)
            if self._llm_thread.is_alive():
                logger.warning("LLM thread did not exit in time")

        # Drain any remaining narration results
        self._drain_narration_results()
        self._cleanup()

    def _handle_scene_cut(self) -> None:
        logger.warning("Scene cut detected! Resetting all tracks.")
        self._id_authority.reset()
        self._small_detector.reset_tracker()
        if self._mid_detector is not None:
            self._mid_detector.reset_tracker()
        self._feature_store.clear()
        self._track_store.clear()
        self._working_memory.reset()
        self._scene_graph.reset()
        self._optical_flow.reset()
        self._predictive_coder.reset()
        # Send reset to LLM thread instead of calling directly
        try:
            self._narration_queue.put_nowait(NarrationRequest(
                deltas=[], key_crops=None, relations_text="",
                memory_text="", screenshot=None,
                frame_id=0, is_reset=True,
            ))
        except queue.Full:
            logger.debug("Narration queue full, reset will be delayed")
        self._rapid_change_count = 0

    def _lazy_load_mid_detector(self) -> None:
        if self._mid_detector is not None:
            return
        # Same model → reuse small detector (preserves tracker state continuity)
        if self._mid_model_name == self._small_model_name:
            logger.info("Mid model same as small (%s), reusing small detector", self._mid_model_name)
            self._mid_detector = self._small_detector
            return
        try:
            logger.info("Lazy-loading mid detector: %s", self._mid_model_name)
            self._mid_detector = YOLODetector(
                self._mid_model_name, self._device,
                self._det_conf, self._det_nms, self._det_imgsz, tier="mid",
                class_whitelist=self._det_class_whitelist,
                min_box_area=self._det_min_box_area,
                tracker_config=self._det_tracker_config,
            )
        except Exception as e:
            logger.warning("Failed to load mid detector, using small: %s", e)

    def _select_key_crops(self, tracking_state) -> list[tuple[int, object]]:
        crops = []
        for eid in tracking_state.new_ids:
            entity = tracking_state.entities.get(eid)
            if entity and entity.crop is not None:
                crops.append((eid, entity.crop))
        if len(crops) < 2:
            active = [
                (eid, e) for eid, e in tracking_state.entities.items()
                if e.is_active and e.crop is not None and eid not in [c[0] for c in crops]
            ]
            active.sort(key=lambda x: x[1].bbox.confidence, reverse=True)
            for eid, e in active:
                if len(crops) >= 2:
                    break
                crops.append((eid, e.crop))
        return crops[:2]

    def _handle_signal(self, signum, frame) -> None:
        logger.info("Received signal %d, stopping...", signum)
        self._running = False
        self._stop_event.set()

    def _cleanup(self) -> None:
        logger.info("Cleaning up...")
        self._analyzer.close()
        logger.info("Pipeline stopped.")


def main():
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    pipeline = Pipeline(config_path)
    pipeline.run()


if __name__ == "__main__":
    main()
