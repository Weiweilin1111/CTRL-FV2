"""CLIP 多模態特徵提取器 —— 二階段解耦架構的第二階段。

第一階段（YOLO-World）負責「框在哪」；本模組對裁切區域計算 CLIP 影像嵌入，
寫入向量索引。查詢時把使用者文字編碼成同空間向量，做跨模態餘弦搜尋：
即使資料庫標籤是 "cell phone"，搜 "mobile"、"手機照片裡那台黑色的" 類查詢
也能靠視覺-語意相似度命中。

優先使用 OpenAI CLIP（YOLO-World set_classes 已快取 ViT-B/32 權重 -> 零下載），
退而求其次 open_clip；都失敗則回傳 NullEmbedder（Null Object 模式，
系統自動降級為字串比對，不會崩潰）。
"""
from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

from ctrlf.config import EmbedderConfig

logger = logging.getLogger("ctrlf.embedder")


class NullEmbedder:
    enabled = False
    dim = 0

    def encode_images(self, crops: Sequence[np.ndarray]) -> np.ndarray | None:
        return None

    def encode_text(self, text: str) -> np.ndarray | None:
        return None


class ClipEmbedder:
    enabled = True

    def __init__(self, model_name: str, device: str):
        import clip  # OpenAI CLIP（ultralytics 相依套件，權重已快取）
        import torch
        self._clip = clip
        self._torch = torch
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        self.dim = int(self.model.visual.output_dim)
        logger.info("CLIP 嵌入器已載入: %s (device=%s, dim=%d)", model_name, device, self.dim)

    def encode_images(self, crops: Sequence[np.ndarray]) -> np.ndarray | None:
        from PIL import Image
        tensors = []
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.preprocess(Image.fromarray(rgb)))
        if not tensors:
            return None
        batch = self._torch.stack(tensors).to(self.device)
        with self._torch.no_grad():
            feats = self.model.encode_image(batch)
        return self._normalize(feats.float().cpu().numpy())

    def encode_text(self, text: str) -> np.ndarray | None:
        text = (text or "").strip()
        if not text:
            return None
        tokens = self._clip.tokenize([text], truncate=True).to(self.device)
        with self._torch.no_grad():
            feats = self.model.encode_text(tokens)
        return self._normalize(feats.float().cpu().numpy())[0]

    @staticmethod
    def _normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norms, 1e-9)


def create_embedder(cfg: EmbedderConfig):
    """工廠：依硬體與環境選擇最佳實作，失敗時優雅降級。"""
    if not cfg.enabled:
        return NullEmbedder()
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return ClipEmbedder(cfg.model, device)
    except Exception as e:  # noqa: BLE001
        logger.warning("CLIP 載入失敗，語意搜尋降級為字串比對: %s", e)
        return NullEmbedder()
