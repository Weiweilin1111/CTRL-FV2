"""VocabularyManager — 統一管理 YOLO-World 的開放詞彙清單。

清單 = 基礎詞彙 ∪ 常駐標籤 ∪ 當前查詢 ∪ 區域錨點。
唯一允許呼叫 detector.set_classes 的地方，避免多處競寫。
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger("ctrlf.vocab")


class VocabularyManager:
    def __init__(self, detector, repo, base_labels: list[str], area_labels: list[str]):
        self._detector = detector
        self._repo = repo
        self._base = [x.strip().lower() for x in base_labels]
        self._areas = [x.strip().lower() for x in area_labels]
        self._query = ""
        self._lock = threading.Lock()

    def refresh(self, query: str | None = None) -> list[str]:
        with self._lock:
            if query is not None:
                self._query = query.strip().lower()
            items = set(self._base)
            items.update(t.strip().lower() for t in self._repo.tags("fixed"))
            items.update(t.strip().lower() for t in self._repo.personal_labels())
            if self._query:
                items.add(self._query)
            items.discard("")
            labels = sorted(items) + self._areas
        self._detector.set_classes(labels)
        logger.info("偵測詞彙已更新（%d 物件 + %d 區域）", len(labels) - len(self._areas), len(self._areas))
        return labels

    @property
    def area_labels(self) -> list[str]:
        return list(self._areas)
