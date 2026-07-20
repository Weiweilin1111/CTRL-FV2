"""IdentityManager — 持續身分引擎：把「框」變成「同一件物品」。

這一層就是視覺記憶的地基：追蹤器的 ID 只活在單次執行，
本引擎以 CLIP 嵌入跨越重啟與消失，維護穩定的 ObjectIdentity。

比對策略（單一閾值在真實 CLIP 分佈上會踩雷，必須三道保險）：
1. 類別閘控 —— 只與同 label 的身分原型比對（cup 永不誤併成 wallet）。
2. EMA 原型 —— 身分原型是觀測的滾動平均（單張定生死太脆弱）。
3. 時空先驗 —— 一小時內同地點再現，門檻放寬 loc_boost
   （剛剛還在書桌上的杯子，大概率還是同一個杯子）。
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime

import numpy as np

from ctrlf.config import MemoryConfig
from ctrlf.storage.sqlite_repo import Repository

logger = logging.getLogger("ctrlf.identity")

_TS_FMT = "%Y-%m-%d %H:%M:%S"


class IdentityManager:
    def __init__(self, repo: Repository, cfg: MemoryConfig):
        self.repo = repo
        self.cfg = cfg
        self._lock = threading.Lock()
        # 內存索引：與 object_identities 表同步
        self._ids: list[str] = []
        self._labels: list[str] = []
        self._matrix: np.ndarray | None = None  # (n, d) 已正規化原型
        self._meta: dict[str, dict] = {}        # object_id -> {display, n_obs, first_seen}
        self._label_seq: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        rows = self.repo.identities()
        protos = []
        expected_dim: int | None = None
        for r in rows:
            if not r["prototype"]:
                continue
            vec = np.frombuffer(r["prototype"], dtype=np.float32)
            if expected_dim is None:
                expected_dim = vec.shape[0]
            elif vec.shape[0] != expected_dim:
                # 換過 CLIP 模型導致維度不一致 -> 略過舊原型，避免 vstack 崩潰
                logger.warning("身分 %s 原型維度 %d != %d，已略過（換過嵌入模型？）",
                               r["display"], vec.shape[0], expected_dim)
                continue
            self._ids.append(r["object_id"])
            self._labels.append(r["label"])
            protos.append(vec / max(float(np.linalg.norm(vec)), 1e-9))
            self._meta[r["object_id"]] = {
                "display": r["display"], "n_obs": r["n_obs"],
                "first_seen": r["first_seen"],
                "last_seen": r["last_seen"], "last_location": r["last_location"],
            }
            base, _, seq = r["display"].rpartition("#")
            if base == r["label"] and seq.isdigit():
                self._label_seq[r["label"]] = max(self._label_seq.get(r["label"], 0), int(seq))
        if protos:
            self._matrix = np.vstack(protos)
        logger.info("身分引擎已載入 %d 個持續身分", len(self._ids))

    def assign(self, vector: np.ndarray, label: str,
               location: str, seen_at: str) -> tuple[str, str]:
        """為一次觀測指派持續身分；回傳 (object_id, display)。"""
        label = (label or "").strip().lower()
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        v = v / max(float(np.linalg.norm(v)), 1e-9)

        with self._lock:
            best_i, best_score = -1, -1.0
            if self._matrix is not None:
                scores = self._matrix @ v
                for i, oid in enumerate(self._ids):
                    if self._labels[i] != label:
                        continue  # 類別閘控
                    if float(scores[i]) > best_score:
                        best_i, best_score = i, float(scores[i])

            threshold = self.cfg.identity_threshold
            if best_i >= 0 and self._recent_same_location(self._ids[best_i], location):
                threshold -= self.cfg.identity_loc_boost  # 時空先驗

            if best_i >= 0 and best_score >= threshold:
                return self._update(best_i, v, location, seen_at)
            return self._create(v, label, location, seen_at)

    # ---- 內部 ----

    def _recent_same_location(self, object_id: str, location: str) -> bool:
        meta = self._meta.get(object_id, {})
        if not location or meta.get("last_location") != location:
            return False
        try:
            last = datetime.strptime(meta.get("last_seen", ""), _TS_FMT)
            return (datetime.now() - last).total_seconds() <= 3600
        except ValueError:
            return False

    def _update(self, idx: int, v: np.ndarray, location: str, seen_at: str) -> tuple[str, str]:
        oid = self._ids[idx]
        alpha = self.cfg.prototype_alpha
        proto = (1 - alpha) * self._matrix[idx] + alpha * v  # EMA 原型
        proto = proto / max(float(np.linalg.norm(proto)), 1e-9)
        self._matrix[idx] = proto

        meta = self._meta[oid]
        meta["n_obs"] += 1
        meta["last_seen"] = seen_at
        meta["last_location"] = location
        self.repo.upsert_identity(
            oid, self._labels[idx], meta["display"], proto.tobytes(),
            meta["n_obs"], meta["first_seen"], seen_at, location,
        )
        return oid, meta["display"]

    def _create(self, v: np.ndarray, label: str, location: str, seen_at: str) -> tuple[str, str]:
        oid = "obj_" + uuid.uuid4().hex[:12]
        seq = self._label_seq.get(label, 0) + 1
        self._label_seq[label] = seq
        display = f"{label}#{seq}"

        self._ids.append(oid)
        self._labels.append(label)
        row = v[None, :]
        self._matrix = row if self._matrix is None else np.vstack([self._matrix, row])
        self._meta[oid] = {"display": display, "n_obs": 1, "first_seen": seen_at,
                           "last_seen": seen_at, "last_location": location}
        self.repo.upsert_identity(oid, label, display, v.tobytes(),
                                  1, seen_at, seen_at, location)
        logger.info("新持續身分: %s（%s @ %s）", display, label, location)
        return oid, display

    def count(self) -> int:
        with self._lock:
            return len(self._ids)

    def prune(self, days: int) -> int:
        """TTL 清理：移除超過保留期未再目擊的身分（記憶體索引 + SQLite）。

        過期判定交給 repo.stale_identity_ids（同時看身分列與目擊列，
        避免誤清持續可見的靜態物體）。目擊紀錄本身有同樣的 TTL——
        過期身分查不出任何情節，只會讓身分矩陣與查詢端點無限膨脹。
        _label_seq 刻意不回收，display 編號（cup#3）不重複使用。
        """
        with self._lock:
            stale_oids = set(self.repo.stale_identity_ids(days))
            if not stale_oids:
                return 0
            keep = [i for i, oid in enumerate(self._ids) if oid not in stale_oids]
            removed = len(self._ids) - len(keep)
            self._ids = [self._ids[i] for i in keep]
            self._labels = [self._labels[i] for i in keep]
            self._matrix = self._matrix[keep] if (self._matrix is not None and keep) else None
            for oid in stale_oids:
                self._meta.pop(oid, None)
            self.repo.delete_identities(list(stale_oids))
        logger.info("身分 TTL 清理：移除 %d 個超過 %d 天未目擊的身分", removed, days)
        return removed
