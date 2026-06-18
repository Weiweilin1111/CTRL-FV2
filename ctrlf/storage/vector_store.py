"""NumpyVectorStore — 內嵌式視覺向量索引（取代 ChromaDB）。

架構決策：
- 舊版 ChromaDB 只拿來存「文字標籤」的句向量，卻拖進 onnxruntime、
  opentelemetry、kubernetes 等十餘個重依賴，並在 API 行程內再載一個
  SentenceTransformer —— 用航空母艦送外賣。
- 尋物場景的向量量級 < 10 萬筆；正規化矩陣內積（精確暴力搜尋）耗時
  在亞毫秒級，比任何 HNSW 服務的 RPC 往返都快，且零部署、零行程。
- 介面遵循 core.interfaces.VectorStore Protocol；未來若多相機/多主機
  擴張，寫一個 QdrantVectorStore 適配器即可無痛替換（DI 容器換一行）。

持久化：matrix -> .npz、ids/payloads -> .json，寫入採 tmp + os.replace 原子替換。
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger("ctrlf.vectors")


class NumpyVectorStore:
    FLUSH_EVERY = 16  # 每累積 N 次寫入就落盤一次

    def __init__(self, base_path: Path):
        self._npz_path = base_path.with_suffix(".npz")
        self._meta_path = base_path.with_suffix(".json")
        base_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ids: list[str] = []
        self._index: dict[str, int] = {}
        self._matrix: np.ndarray | None = None  # (n, d) 已正規化
        self._payloads: dict[str, dict] = {}
        self._dirty = 0
        self._load()

    def _load(self) -> None:
        try:
            if self._npz_path.exists() and self._meta_path.exists():
                matrix = np.load(self._npz_path)["matrix"].astype(np.float32)
                meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
                ids = meta["ids"]
                # 空索引（或形狀不符）一律視為全新狀態，避免載入 (0, d) 毒矩陣
                if matrix.ndim == 2 and matrix.shape[0] == len(ids) and len(ids) > 0:
                    self._ids = ids
                    self._matrix = matrix
                    self._payloads = meta["payloads"]
                    self._index = {vid: i for i, vid in enumerate(ids)}
                    logger.info("向量索引已載入 %d 筆", len(ids))
        except Exception as e:  # noqa: BLE001 — 索引可重建，損毀時直接重新開始
            logger.warning("向量索引載入失敗，重新開始: %s", e)
            self._ids, self._index, self._matrix, self._payloads = [], {}, None, {}

    def upsert(self, vid: str, vector: np.ndarray, payload: dict) -> None:
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        v = v / max(float(np.linalg.norm(v)), 1e-9)
        with self._lock:
            if vid in self._index:
                self._matrix[self._index[vid]] = v
            else:
                # 先算新矩陣再登記 id，任何一步失敗都不會留下不一致狀態
                row = v[None, :]
                if self._matrix is None or self._matrix.shape[0] == 0:
                    new_matrix = row
                elif self._matrix.shape[1] != row.shape[1]:
                    # 嵌入維度改變（換了 CLIP 模型）-> 舊向量不相容，重建索引
                    logger.warning("向量維度由 %d 變為 %d，索引已重建",
                                   self._matrix.shape[1], row.shape[1])
                    self._ids, self._index, self._payloads = [], {}, {}
                    new_matrix = row
                else:
                    new_matrix = np.vstack([self._matrix, row])
                self._matrix = new_matrix
                self._index[vid] = len(self._ids)
                self._ids.append(vid)
            self._payloads[vid] = dict(payload)
            self._dirty += 1
            if self._dirty >= self.FLUSH_EVERY:
                self._flush_locked()

    def search(self, vector: np.ndarray, k: int = 6, min_score: float = 0.0) -> list[dict]:
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        v = v / max(float(np.linalg.norm(v)), 1e-9)
        with self._lock:
            if self._matrix is None or self._matrix.shape[0] == 0 or len(self._ids) == 0:
                return []
            scores = self._matrix @ v
            order = np.argsort(-scores)[: max(1, k)]
            results = []
            for i in order:
                score = float(scores[i])
                if score < min_score:
                    break
                vid = self._ids[i]
                results.append({"id": vid, "score": round(score, 4),
                                "payload": dict(self._payloads.get(vid, {}))})
            return results

    def delete(self, ids: Sequence[str]) -> None:
        targets = {i for i in ids if i in self._index}
        if not targets:
            return
        with self._lock:
            keep = [i for i, vid in enumerate(self._ids) if vid not in targets]
            self._ids = [self._ids[i] for i in keep]
            self._matrix = self._matrix[keep] if (self._matrix is not None and keep) else None
            for vid in targets:
                self._payloads.pop(vid, None)
            self._index = {vid: i for i, vid in enumerate(self._ids)}
            self._dirty += 1
            self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def count(self) -> int:
        with self._lock:
            return len(self._ids)

    def _flush_locked(self) -> None:
        try:
            # 注意：np.savez 會對非 .npz 結尾的檔名自動追加 .npz，暫存檔必須以 .npz 結尾
            tmp_npz = self._npz_path.with_name(self._npz_path.stem + ".tmp.npz")
            matrix = self._matrix if self._matrix is not None else np.zeros((0, 1), np.float32)
            np.savez_compressed(tmp_npz, matrix=matrix)
            os.replace(tmp_npz, self._npz_path)

            tmp_meta = self._meta_path.with_suffix(".json.tmp")
            tmp_meta.write_text(
                json.dumps({"ids": self._ids, "payloads": self._payloads}, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp_meta, self._meta_path)
            self._dirty = 0
        except Exception as e:  # noqa: BLE001
            logger.warning("向量索引落盤失敗: %s", e)
