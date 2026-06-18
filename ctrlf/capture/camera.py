"""CaptureService — 獨立擷取執行緒（生產者）。

只做一件事：以相機原生速率讀取影格並發佈到 StateHub。
不做推論、不做編碼、不碰資料庫；斷線自動重連。
"""
from __future__ import annotations

import logging
import sys
import threading
import time

import cv2

from ctrlf.config import CameraConfig
from ctrlf.runtime.hub import StateHub

logger = logging.getLogger("ctrlf.capture")


class CaptureService(threading.Thread):
    def __init__(self, cfg: CameraConfig, hub: StateHub):
        super().__init__(name="ctrlf-capture", daemon=True)
        self.cfg = cfg
        self.hub = hub
        self._stop_event = threading.Event()
        self._cap: cv2.VideoCapture | None = None

    def run(self) -> None:
        fail_streak = 0
        while not self._stop_event.is_set():
            if self._cap is None:
                if not self._open():
                    time.sleep(2.0)
                    continue
            ok, frame = self._cap.read()
            if not ok or frame is None:
                fail_streak += 1
                if fail_streak >= 10:
                    logger.warning("相機連續讀取失敗，嘗試重新開啟…")
                    self._release()
                    fail_streak = 0
                time.sleep(0.05)
                continue
            fail_streak = 0
            self.hub.publish_frame(frame)
        self._release()
        logger.info("CaptureService 已停止")

    def _open(self) -> bool:
        backend = cv2.CAP_DSHOW if (self.cfg.use_dshow and sys.platform == "win32") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.cfg.index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg.fourcc))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 信箱模式：不要讓驅動囤積舊幀
        if not cap.isOpened():
            cap.release()
            logger.error("無法開啟相機 index=%d", self.cfg.index)
            return False
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("相機已開啟 index=%d 解析度=%dx%d", self.cfg.index, w, h)
        self._cap = cap
        return True

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def stop(self) -> None:
        self._stop_event.set()
