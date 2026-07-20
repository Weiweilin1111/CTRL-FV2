"""影像降載雙保險：ROI 邊界遮罩 + 像素差分動態閘控。

從源頭切斷硬體過載：
1. RoiMask  — 偵測範圍外的像素直接抹黑，背景雜訊進不了模型。
2. MotionGate — 畫面靜止時完全不推論；只有動態超過門檻、或到達
   force_interval 的定時抽樣點，才放行一次推論。再疊一層
   max_infer_fps 頻率硬上限，雜亂場景下也不可能吃滿 GPU。
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from ctrlf.config import MotionConfig, RoiConfig


class RoiMask:
    def __init__(self, cfg: RoiConfig):
        self.cfg = cfg
        self._mask: np.ndarray | None = None
        self._size: tuple[int, int] | None = None

    @property
    def active(self) -> bool:
        if not self.cfg.enabled:
            return False
        x1, y1, x2, y2 = self.cfg.rect
        return not (x1 <= 0.0 and y1 <= 0.0 and x2 >= 1.0 and y2 >= 1.0)

    @property
    def area_fraction(self) -> float:
        """ROI 佔全畫面的比例，用來校正動態比率的分母。"""
        if not self.active:
            return 1.0
        x1, y1, x2, y2 = self.cfg.rect
        return max(1e-3, min(1.0, (x2 - x1) * (y2 - y1)))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.active:
            return frame
        h, w = frame.shape[:2]
        if self._mask is None or self._size != (w, h):
            x1, y1, x2, y2 = self.cfg.rect
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = 255
            self._mask = mask
            self._size = (w, h)
        return cv2.bitwise_and(frame, frame, mask=self._mask)


class MotionGate:
    def __init__(self, cfg: MotionConfig):
        self.cfg = cfg
        self._bg: np.ndarray | None = None
        self._last_infer = 0.0
        self._frames = 0

    def rate_ok(self, now: float, max_fps: float) -> bool:
        """頻率硬上限：未到最小間隔時連差分都不算，純零成本跳過。"""
        return (now - self._last_infer) >= 1.0 / max(0.5, max_fps)

    def evaluate(self, frame: np.ndarray, now: float, sensitivity: float,
                 roi_fraction: float = 1.0) -> tuple[bool, float]:
        """回傳 (是否觸發推論, 動態像素比率)。"""
        h, w = frame.shape[:2]
        sw = self.cfg.downscale_width
        sh = max(1, int(h * sw / max(1, w)))
        gray = cv2.cvtColor(cv2.resize(frame, (sw, sh)), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._bg is None:
            self._bg = gray.astype(np.float32)
            self._last_infer = now
            self._frames += 1
            return True, 1.0

        diff = cv2.absdiff(gray, cv2.convertScaleAbs(self._bg))
        cv2.accumulateWeighted(gray.astype(np.float32), self._bg, 0.06)
        _, thresh = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
        ratio = float(cv2.countNonZero(thresh)) / float(thresh.size) / max(roi_fraction, 1e-6)

        # 靈敏度 1.0 -> 門檻 0.2% 動態像素即觸發；0.0 -> 需 5.2%
        threshold = 0.002 + (1.0 - max(0.0, min(1.0, sensitivity))) * 0.05
        force = (now - self._last_infer) >= self.cfg.force_interval
        trigger = (self._frames <= self.cfg.warmup_frames) or force or (ratio >= threshold)

        self._frames += 1
        if trigger:
            self._last_infer = now
        return trigger, round(ratio, 4)
