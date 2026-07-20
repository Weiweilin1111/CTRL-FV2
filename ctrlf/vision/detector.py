"""YoloWorldDetector — 開放詞彙偵測器（Strategy 模式的預設實作）。

效能策略：
- 預設載入輕量 yolov8s-worldv2（首選），保底退回本地既有的大模型。
- CUDA 上以 FP16 推論、imgsz 由 960 降為 640、max_det 截斷，
  搭配上游 MotionGate，VRAM 與延遲都大幅下降。
- 內建鎖：set_classes（重算文字嵌入）與 detect 不會同時撕裂模型狀態。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from ctrlf.config import ROOT_DIR, DetectorConfig
from ctrlf.core.models import Detection

logger = logging.getLogger("ctrlf.detector")


class YoloWorldDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        if cfg.device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device
        self.half = bool(cfg.half) and self.device != "cpu"
        self.model, self.model_name = self._load()
        if self.device != "cpu":
            self.model.to(self.device)
        self._pin_text_model()

    def _pin_text_model(self) -> None:
        """把 set_classes 用的 CLIP 文字模型固定建在 CPU，並掛成「普通屬性」。

        ultralytics 預設把它快取成 WorldModel 的子模組，predict(half=True)
        會原地把整棵模組樹轉 FP16、.to() 也會搬移它，導致執行期 set_classes
        出現 device/dtype 不一致而崩潰。換詞彙是低頻操作，CPU 編碼十幾個
        詞僅需數十毫秒，還省下一份 VRAM。
        """
        try:
            from ultralytics.nn.text_model import build_text_model
            world_model = self.model.model
            text_model = build_text_model("clip:ViT-B/32", device=torch.device("cpu"))
            world_model._modules.pop("clip_model", None)
            object.__setattr__(world_model, "clip_model", text_model)  # 繞過 nn.Module 註冊
            logger.info("CLIP 文字模型已固定於 CPU（隔離於 half()/to() 之外）")
        except Exception as e:  # noqa: BLE001 — 失敗則回退 ultralytics 預設行為
            logger.warning("文字模型預建失敗，回退預設行為: %s", e)

    def _load(self) -> tuple[YOLO, str]:
        errors = []
        for cand in self.cfg.model_candidates:
            p = Path(cand)
            local = p if p.is_absolute() else ROOT_DIR / cand
            source = str(local) if local.exists() else cand  # 不在本地則交給 ultralytics 下載
            try:
                model = YOLO(source)
                logger.info("偵測模型已載入: %s (device=%s, half=%s)", source, self.device, self.half)
                return model, Path(source).name
            except Exception as e:  # noqa: BLE001 — 逐一嘗試候選模型
                errors.append(f"{cand}: {e}")
                logger.warning("載入 %s 失敗，嘗試下一個候選…", cand)
        raise RuntimeError("無法載入任何偵測模型:\n" + "\n".join(errors))

    def set_classes(self, labels) -> None:
        labels = [x for x in dict.fromkeys(str(s).strip().lower() for s in labels) if x]
        if not labels:
            return
        with self._lock:
            self.model.set_classes(labels)

    def detect(self, frame: np.ndarray, conf: float) -> list[Detection]:
        with self._lock:
            results = self.model.predict(
                frame,
                imgsz=self.cfg.imgsz,
                conf=conf,
                iou=self.cfg.iou,
                max_det=self.cfg.max_det,
                half=self.half,
                device=self.device,
                agnostic_nms=True,
                verbose=False,
            )
        result = results[0]
        detections: list[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        h, w = frame.shape[:2]
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clses = result.boxes.cls.cpu().numpy()
        for box, c, cls_id in zip(xyxy, confs, clses):
            x1 = int(max(0, min(w - 1, box[0])))
            y1 = int(max(0, min(h - 1, box[1])))
            x2 = int(max(0, min(w, box[2])))
            y2 = int(max(0, min(h, box[3])))
            if x2 <= x1 or y2 <= y1:
                continue
            label = str(result.names[int(cls_id)]).lower()
            detections.append(Detection(label=label, confidence=float(c), bbox=(x1, y1, x2, y2)))
        return detections

    def warmup(self) -> None:
        dummy = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
        try:
            self.detect(dummy, conf=0.5)
            logger.info("偵測模型暖機完成")
        except Exception as e:  # noqa: BLE001
            logger.warning("暖機失敗（不影響啟動）: %s", e)

    @property
    def info(self) -> dict:
        return {
            "model": self.model_name,
            "device": "cuda" if str(self.device).startswith("cuda") else "cpu",
            "imgsz": self.cfg.imgsz,
            "half": self.half,
        }
