"""PersonalItemService — 「我的東西」照片註冊、逐張管理與背景比對。

使用者上傳 3~5 張自己物品的照片 → CLIP 影像嵌入存入獨立向量索引
（personal_vectors，與歷史目擊索引分離），每張參考照可獨立檢視/刪除/補充。
EmbeddingWorker 每算出一個即時裁切嵌入，就順手與註冊錨點比對；
命中即更新該物品的「最後行蹤」（位置/時間/相似度/縮圖）。

精度規則：偵測類別與註冊類別一致時用基本門檻；類別不一致時
門檻提高（+0.08），避免「同類不同物」的誤認。
test_match 提供註冊品質檢測：上傳一張現場照，回報最高相似度。
"""
from __future__ import annotations

import logging
import threading
import uuid

import cv2
import numpy as np

from ctrlf.core.interfaces import Embedder, VectorStore
from ctrlf.storage.sqlite_repo import Repository

logger = logging.getLogger("ctrlf.personal")


class PersonalItemService:
    CROSS_LABEL_PENALTY = 0.08  # 類別不一致時的額外門檻

    def __init__(self, repo: Repository, store: VectorStore, embedder: Embedder,
                 min_score: float, thumb_size: int = 160):
        self.repo = repo
        self.store = store
        self.embedder = embedder
        self.min_score = float(min_score)
        self.thumb_size = thumb_size
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.embedder.enabled

    # ---- 註冊與照片管理 ----

    def register(self, name: str, label: str, images: list[np.ndarray]) -> dict | None:
        """以多張參考照註冊一件個人物品；CLIP 不可用或全部編碼失敗回 None。"""
        if not self.available or not images:
            return None
        item_id = uuid.uuid4().hex
        label = label.strip().lower()
        self.repo.add_personal_item(item_id, name.strip(), label, 0, self._thumb(images[0]))
        added = self.add_photos(item_id, images)
        if added == 0:
            self.repo.delete_personal_item(item_id)
            return None
        logger.info("個人物品已註冊: %s (%s, %d 張參考照)", name, label, added)
        return {"item_id": item_id, "name": name, "label": label, "n_photos": added}

    def add_photos(self, item_id: str, images: list[np.ndarray]) -> int:
        """為既有物品補充參考照；回傳成功加入的張數。"""
        item = self.repo.get_personal_item(item_id)
        if item is None or not self.available:
            return 0
        images = [img for img in images if img is not None and img.size > 0]
        if not images:
            return 0
        vectors = self.embedder.encode_images(images)
        if vectors is None or len(vectors) != len(images):
            return 0

        payload = {"item_id": item_id, "name": item["name"], "label": item["label"]}
        rows: list[tuple[str, bytes | None]] = []
        with self._lock:
            for img, vec in zip(images, vectors):
                photo_id = uuid.uuid4().hex
                self.store.upsert(f"{item_id}#{photo_id}", vec, dict(payload))
                rows.append((photo_id, self._thumb(img)))
            self.store.flush()
        self.repo.add_personal_photos(item_id, rows)
        return len(rows)

    def remove_photo(self, item_id: str, photo_id: str) -> None:
        with self._lock:
            self.store.delete([f"{item_id}#{photo_id}"])
            self.store.flush()
        self.repo.delete_personal_photo(item_id, photo_id)

    def delete(self, item_id: str) -> bool:
        if self.repo.get_personal_item(item_id) is None:
            return False
        photo_ids = [p["photo_id"] for p in self.repo.personal_photos(item_id)]
        with self._lock:
            self.store.delete([f"{item_id}#{pid}" for pid in photo_ids])
            self.store.flush()
        self.repo.delete_personal_item(item_id)
        return True

    # ---- 比對 ----

    def match(self, vector: np.ndarray, label: str, location: str,
              seen_at: str, crop: np.ndarray | None) -> None:
        """由 EmbeddingWorker 對每個即時裁切呼叫；命中則更新物品最後行蹤。"""
        if self.store.count() == 0:
            return
        for hit in self.store.search(vector, k=3, min_score=self.min_score):
            payload = hit["payload"]
            required = self.min_score
            if payload.get("label") != (label or "").strip().lower():
                required += self.CROSS_LABEL_PENALTY
            if hit["score"] < required:
                continue
            thumb = self._thumb(crop) if crop is not None else None
            self.repo.update_personal_sighting(
                payload["item_id"], location, seen_at, hit["score"], thumb
            )
            logger.info("個人物品命中: %s @ %s (相似度 %.2f)",
                        payload.get("name"), location, hit["score"])
            return  # 一個裁切最多歸屬一件物品

    def test_match(self, item_id: str, image: np.ndarray) -> float | None:
        """註冊品質檢測：回傳測試照與該物品參考照的最高相似度。"""
        if not self.available or image is None or image.size == 0:
            return None
        vectors = self.embedder.encode_images([image])
        if vectors is None:
            return None
        hits = self.store.search(vectors[0], k=max(1, self.store.count()), min_score=0.0)
        scores = [h["score"] for h in hits if h["payload"].get("item_id") == item_id]
        return max(scores) if scores else None

    def _thumb(self, image: np.ndarray) -> bytes | None:
        if image is None or image.size == 0:
            return None
        h, w = image.shape[:2]
        scale = self.thumb_size / max(h, w)
        if scale < 1.0:
            image = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return buf.tobytes() if ok else None
