"""StateHub — 全系統的執行緒安全狀態中樞（Producer-Consumer 的交會點）。

設計核心：
- 「最新影格信箱」(latest-frame mailbox)：擷取執行緒永遠覆寫最新一張，
  推論執行緒醒來只拿最新的，天生防止佇列堆積與延遲累積。
- 串流端點只讀快照，與推論完全脫鉤：推論 3 FPS 時畫面依然能跑滿 24 FPS。
- JPEG 編碼結果以 (frame_id, results_version, settings_version) 為鍵快取，
  多個瀏覽器分頁同時觀看也只編碼一次。
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import cv2
import numpy as np

from ctrlf.core.models import FramePacket


class RateMeter:
    """指數移動平均的頻率計。"""

    def __init__(self, alpha: float = 0.15):
        self._alpha = alpha
        self._last: float | None = None
        self._fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                self._fps = inst if self._fps == 0.0 else (
                    (1 - self._alpha) * self._fps + self._alpha * inst
                )
        self._last = now
        return round(self._fps, 1)


class StateHub:
    def __init__(self):
        self._cond = threading.Condition()
        self._packet: FramePacket | None = None
        self._frame_seq = 0
        self._tracks: list[dict] = []
        self._results_version = 0

        self._stats_lock = threading.Lock()
        self._stats: dict = {
            "capture_fps": 0.0, "infer_fps": 0.0, "infer_ms": 0.0,
            "motion_ratio": 0.0, "detections": 0,
        }
        self._cap_meter = RateMeter()

        self._jpeg_lock = threading.Lock()
        self._jpeg_cache: tuple[tuple, bytes] | None = None

    # ---- 影格（生產者：CaptureService / 消費者：InferencePipeline、串流） ----

    def publish_frame(self, frame: np.ndarray, ts: float | None = None) -> None:
        with self._cond:
            self._frame_seq += 1
            self._packet = FramePacket(frame, self._frame_seq, ts or time.time())
            self._cond.notify_all()
        self.update_stats(capture_fps=self._cap_meter.tick())

    def wait_frame(self, last_id: int, timeout: float = 0.5) -> FramePacket | None:
        """阻塞等待比 last_id 更新的影格；逾時回傳 None。"""
        with self._cond:
            if self._packet is None or self._packet.frame_id <= last_id:
                self._cond.wait(timeout)
            pkt = self._packet
            if pkt is not None and pkt.frame_id > last_id:
                return pkt
            return None

    def latest_frame(self) -> FramePacket | None:
        with self._cond:
            return self._packet

    # ---- 偵測結果與統計 ----

    def publish_results(self, tracks: list[dict]) -> None:
        with self._cond:
            self._tracks = tracks
            self._results_version += 1

    def update_stats(self, **kwargs) -> None:
        with self._stats_lock:
            self._stats.update(kwargs)

    def snapshot(self) -> dict:
        with self._cond:
            tracks = list(self._tracks)
            version = self._results_version
        with self._stats_lock:
            stats = dict(self._stats)
        return {"tracks": tracks, "stats": stats, "version": version}

    # ---- 串流渲染（含編碼快取） ----

    def render_jpeg(
        self,
        draw_fn: Callable[[np.ndarray, list[dict]], np.ndarray],
        settings_version: int,
        quality: int = 80,
    ) -> bytes | None:
        with self._cond:
            pkt = self._packet
            tracks = list(self._tracks)
            results_version = self._results_version
        if pkt is None:
            return None

        key = (pkt.frame_id, results_version, settings_version)
        # 繪製與編碼整段在鎖內序列化：多客戶端同時 cache miss 時，
        # 第二個等第一個編完直接拿快取，不再各編一份（編碼僅數 ms）
        with self._jpeg_lock:
            if self._jpeg_cache is not None and self._jpeg_cache[0] == key:
                return self._jpeg_cache[1]
            annotated = draw_fn(pkt.frame, tracks)
            ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                return None
            data = buf.tobytes()
            self._jpeg_cache = (key, data)
            return data
