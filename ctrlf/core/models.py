"""模組間傳遞的核心資料結構 —— 唯一的共享語言，避免模組互相 import 實作。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class Detection:
    """單次偵測結果（像素座標）。"""
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass(frozen=True, slots=True)
class FramePacket:
    """擷取執行緒發佈的最新影格（mailbox 模式：永遠只留最新一張）。"""
    frame: np.ndarray
    frame_id: int
    ts: float


@dataclass(slots=True)
class SightingEvent:
    """追蹤器產生的「目擊事件」，由 StorageWriter 批次寫入 SQLite。"""
    sighting_id: str
    label: str
    instance: int
    location: str
    bbox: tuple[int, int, int, int]
    confidence: float
    first_seen: str   # "YYYY-MM-DD HH:MM:SS"
    last_seen: str
    thumb_jpeg: bytes | None = None  # 縮圖只在確認/大幅移動時更新

    @property
    def display_name(self) -> str:
        return f"{self.label}-{self.instance}"


@dataclass(slots=True)
class EmbedJob:
    """交給 EmbeddingWorker 的 CLIP 編碼任務。"""
    sighting_id: str
    payload: dict = field(default_factory=dict)
    crop: np.ndarray | None = None
