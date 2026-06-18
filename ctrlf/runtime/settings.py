"""執行期可調參數的執行緒安全容器 —— UI 滑桿改動即時生效，不需重啟。"""
from __future__ import annotations

import threading

from ctrlf.config import AppConfig

_CLAMPS = {
    "conf": (0.01, 0.9),
    "motion_sensitivity": (0.0, 1.0),
    "max_infer_fps": (0.5, 15.0),
}


class RuntimeSettings:
    def __init__(self, cfg: AppConfig):
        self._lock = threading.Lock()
        self._data: dict = {
            "conf": cfg.detector.conf,
            "motion_sensitivity": cfg.motion.sensitivity,
            "max_infer_fps": cfg.motion.max_infer_fps,
            "show_all": False,
            "query": "",
        }
        self._version = 0

    def get(self, key: str):
        with self._lock:
            return self._data[key]

    def snapshot(self) -> dict:
        with self._lock:
            return {**self._data, "version": self._version}

    def update(self, **kwargs) -> dict:
        with self._lock:
            for key, value in kwargs.items():
                if value is None or key not in self._data:
                    continue
                if key in _CLAMPS:
                    lo, hi = _CLAMPS[key]
                    value = max(lo, min(hi, float(value)))
                if key == "query":
                    value = str(value).strip().lower()
                if key == "show_all":
                    value = bool(value)
                self._data[key] = value
            self._version += 1
            return {**self._data, "version": self._version}

    @property
    def version(self) -> int:
        with self._lock:
            return self._version
