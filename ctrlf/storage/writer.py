"""StorageWriter — 批次寫入執行緒：推論丟事件進佇列即返回，永不阻塞。

改造重點（對比舊版 _db_worker_loop）：
- 批次 executemany + 單次 commit，高頻偵測下寫入放大降一個數量級。
- 不再假裝 SQLite 與向量庫是同一筆交易（舊版的「雙軌防護」會在
  Chroma 成功後 rollback SQLite，反而製造孤兒）；改為 SQLite 為
  事實來源，向量索引可隨時由縮圖重建，週期性 TTL 同步刪除。
"""
from __future__ import annotations

import logging
import queue
import threading
import time

from ctrlf.core.interfaces import VectorStore
from ctrlf.core.models import SightingEvent
from ctrlf.storage.sqlite_repo import Repository

logger = logging.getLogger("ctrlf.writer")


class StorageWriter(threading.Thread):
    BATCH = 64
    CLEANUP_INTERVAL = 3600.0

    def __init__(self, repo: Repository, vectors: VectorStore, retention_days: int):
        super().__init__(name="ctrlf-writer", daemon=True)
        self.repo = repo
        self.vectors = vectors
        self.retention_days = retention_days
        self._q: queue.Queue = queue.Queue(maxsize=2048)
        self._stop_event = threading.Event()
        self._dropped = 0
        self._last_cleanup = time.time()

    def enqueue(self, event: SightingEvent) -> None:
        try:
            self._q.put_nowait(event)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 100 == 1:
                logger.warning("寫入佇列滿載，已累計丟棄 %d 筆事件", self._dropped)

    def run(self) -> None:
        while not (self._stop_event.is_set() and self._q.empty()):
            batch: list[SightingEvent] = []
            try:
                batch.append(self._q.get(timeout=0.5))
            except queue.Empty:
                pass
            while batch and len(batch) < self.BATCH:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            if batch:
                try:
                    self.repo.upsert_sightings(batch)
                except Exception as e:  # noqa: BLE001
                    logger.exception("批次寫入失敗（%d 筆）: %s", len(batch), e)

            now = time.time()
            if now - self._last_cleanup >= self.CLEANUP_INTERVAL:
                self._last_cleanup = now
                try:
                    removed = self.repo.cleanup(self.retention_days)
                    if removed:
                        self.vectors.delete(removed)
                    self.vectors.flush()
                except Exception as e:  # noqa: BLE001
                    logger.exception("週期清理失敗: %s", e)
        logger.info("StorageWriter 已停止（剩餘佇列已清空）")

    def stop(self) -> None:
        self._stop_event.set()
