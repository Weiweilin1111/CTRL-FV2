"""InferencePipeline / EmbeddingWorker — 消費者執行緒。

資料流（完全解耦的 Producer-Consumer）：
  CaptureService ──> StateHub(最新影格信箱) ──> InferencePipeline
                                                   ├─> StateHub(軌跡快照, 供串流/WS)
                                                   ├─> StorageWriter 佇列(SQLite)
                                                   └─> EmbedQueue ──> EmbeddingWorker ──> VectorStore
推論執行緒永不寫磁碟、永不等待網路；寫入端永不阻塞推論。
"""
from __future__ import annotations

import logging
import queue
import threading
import time

from ctrlf.capture.motion import MotionGate, RoiMask
from ctrlf.core.interfaces import Detector, Embedder, VectorStore
from ctrlf.core.models import EmbedJob
from ctrlf.runtime.hub import RateMeter, StateHub
from ctrlf.runtime.settings import RuntimeSettings
from ctrlf.vision.tracker import IouTracker

logger = logging.getLogger("ctrlf.pipeline")


class InferencePipeline(threading.Thread):
    def __init__(
        self,
        hub: StateHub,
        detector: Detector,
        tracker: IouTracker,
        gate: MotionGate,
        roi: RoiMask,
        settings: RuntimeSettings,
        writer,                      # StorageWriter
        embed_queue: queue.Queue,
        area_labels: list[str],
        embed_enabled: bool,
    ):
        super().__init__(name="ctrlf-pipeline", daemon=True)
        self.hub = hub
        self.detector = detector
        self.tracker = tracker
        self.gate = gate
        self.roi = roi
        self.settings = settings
        self.writer = writer
        self.embed_queue = embed_queue
        self.embed_enabled = embed_enabled
        self._area_set = {x.strip().lower() for x in area_labels}
        self._stop_event = threading.Event()

    def run(self) -> None:
        meter = RateMeter()
        last_id = 0
        while not self._stop_event.is_set():
            pkt = self.hub.wait_frame(last_id, timeout=0.5)
            if pkt is None:
                continue
            last_id = pkt.frame_id
            now = time.time()
            s = self.settings.snapshot()

            # 第一道閘門：推論頻率硬上限（零成本跳過）
            if not self.gate.rate_ok(now, s["max_infer_fps"]):
                continue

            # 第二道閘門：ROI 遮罩 + 像素差分動態觸發
            masked = self.roi.apply(pkt.frame)
            trigger, ratio = self.gate.evaluate(
                masked, now, s["motion_sensitivity"], self.roi.area_fraction
            )
            self.hub.update_stats(motion_ratio=ratio)
            if not trigger:
                continue

            t0 = time.perf_counter()
            try:
                detections = self.detector.detect(masked, conf=s["conf"])
            except Exception as e:  # noqa: BLE001 — 單幀失敗不可拖垮管線
                logger.exception("推論失敗（跳過此幀）: %s", e)
                continue
            latency_ms = (time.perf_counter() - t0) * 1000.0

            items = [d for d in detections if d.label not in self._area_set]
            areas = [d for d in detections if d.label in self._area_set]
            snapshots, events, embed_jobs = self.tracker.update(items, areas, pkt.frame, now)

            self.hub.publish_results(snapshots)
            self.hub.update_stats(
                infer_fps=meter.tick(),
                infer_ms=round(latency_ms, 1),
                detections=len(items),
            )

            for ev in events:
                self.writer.enqueue(ev)
            if self.embed_enabled:
                for job in embed_jobs:
                    try:
                        self.embed_queue.put_nowait(job)
                    except queue.Full:
                        pass  # 嵌入是輔助功能，滿了就丟，不能堵塞推論
        logger.info("InferencePipeline 已停止")

    def stop(self) -> None:
        self._stop_event.set()


class EmbeddingWorker(threading.Thread):
    """低優先序消費者：批次計算 CLIP 影像嵌入並寫入向量索引。"""

    def __init__(self, embedder: Embedder, store: VectorStore,
                 jobs: queue.Queue, batch_size: int = 8, personal=None,
                 identity=None, repo=None):
        super().__init__(name="ctrlf-embed", daemon=True)
        self.embedder = embedder
        self.store = store
        self.jobs = jobs
        self.batch_size = max(1, batch_size)
        self.personal = personal    # PersonalItemService | None
        self.identity = identity    # IdentityManager | None（持續身分指派）
        self.repo = repo            # Repository | None
        # 目擊列由 StorageWriter 批次寫入（~0.5s），嵌入可能先算完 ->
        # set_sighting_object 撲空時暫存於此，下一輪重試（防身分漏掛）
        self._pending_assign: list[tuple[str, str, int]] = []
        self._stop_event = threading.Event()

    def _flush_pending_assign(self) -> None:
        if not self._pending_assign or self.repo is None:
            return
        still_pending = []
        for sighting_id, object_id, attempts in self._pending_assign:
            if not self.repo.set_sighting_object(sighting_id, object_id) and attempts < 10:
                still_pending.append((sighting_id, object_id, attempts + 1))
        self._pending_assign = still_pending

    def run(self) -> None:
        while not self._stop_event.is_set():
            self._flush_pending_assign()
            try:
                first: EmbedJob = self.jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            batch = [first]
            while len(batch) < self.batch_size:
                try:
                    batch.append(self.jobs.get_nowait())
                except queue.Empty:
                    break
            # 編碼器會默默跳過空裁切，導致 zip 對位錯亂（向量掛到別的目擊上）
            # —— 先過濾再編碼，並以長度檢查兜底
            batch = [j for j in batch if j.crop is not None and j.crop.size > 0]
            if not batch:
                continue
            try:
                vectors = self.embedder.encode_images([j.crop for j in batch])
                if vectors is None:
                    continue
                if len(vectors) != len(batch):
                    logger.warning("嵌入批次數量不符（%d 圖 -> %d 向量），整批丟棄以防錯掛",
                                   len(batch), len(vectors))
                    continue
                for job, vec in zip(batch, vectors):
                    self.store.upsert(job.sighting_id, vec, job.payload)
                    if self.identity is not None and self.repo is not None:
                        # 持續身分指派：框 -> 同一件物品（視覺記憶的地基）
                        object_id, _ = self.identity.assign(
                            vec,
                            job.payload.get("label", ""),
                            job.payload.get("location", "unknown"),
                            job.payload.get("last_seen", ""),
                        )
                        if not self.repo.set_sighting_object(job.sighting_id, object_id):
                            # 目擊列尚未被 StorageWriter 寫入，排入重試
                            self._pending_assign.append((job.sighting_id, object_id, 0))
                    if self.personal is not None:  # 順手與「我的物品」錨點比對
                        self.personal.match(
                            vec,
                            job.payload.get("label", ""),
                            job.payload.get("location", "unknown"),
                            job.payload.get("last_seen", ""),
                            job.crop,
                        )
            except Exception as e:  # noqa: BLE001
                logger.exception("嵌入批次失敗: %s", e)
        logger.info("EmbeddingWorker 已停止")

    def stop(self) -> None:
        self._stop_event.set()
