"""AppContainer — 依賴注入的組裝根（Composition Root）。

整個系統只有這裡 new 具體類別；其他模組只依賴建構子參數與 Protocol。
要換偵測器、向量庫、嵌入器，改這一個檔案即可。
"""
from __future__ import annotations

import logging
import queue

from ctrlf.capture.camera import CaptureService
from ctrlf.capture.motion import MotionGate, RoiMask
from ctrlf.config import ROOT_DIR, AppConfig
from ctrlf.runtime.hub import StateHub
from ctrlf.runtime.identity import IdentityManager
from ctrlf.runtime.lexicon import Lexicon
from ctrlf.runtime.memory import MemoryService
from ctrlf.runtime.personal import PersonalItemService
from ctrlf.runtime.settings import RuntimeSettings
from ctrlf.runtime.vocabulary import VocabularyManager
from ctrlf.storage.sqlite_repo import Repository
from ctrlf.storage.vector_store import NumpyVectorStore
from ctrlf.storage.writer import StorageWriter
from ctrlf.vision.annotator import Annotator
from ctrlf.vision.detector import YoloWorldDetector
from ctrlf.vision.embedder import create_embedder
from ctrlf.vision.pipeline import EmbeddingWorker, InferencePipeline
from ctrlf.vision.tracker import IouTracker

logger = logging.getLogger("ctrlf.container")


class AppContainer:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        data_dir = ROOT_DIR / cfg.storage.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        # 共享狀態
        self.hub = StateHub()
        self.settings = RuntimeSettings(cfg)

        # 儲存層
        self.repo = Repository(data_dir / "ctrlf.db")
        self.vectors = NumpyVectorStore(data_dir / "vectors")
        self.personal_vectors = NumpyVectorStore(data_dir / "personal_vectors")

        # 中文/別名翻譯層
        self.lexicon = Lexicon(self.repo.aliases())

        # AI 核心
        self.detector = YoloWorldDetector(cfg.detector)
        self.embedder = create_embedder(cfg.embedder)
        self.tracker = IouTracker(cfg.tracker, cfg.storage.thumb_size)
        self.annotator = Annotator()
        self.vocab = VocabularyManager(
            self.detector, self.repo, cfg.detector.base_labels, cfg.detector.area_labels
        )
        self.personal = PersonalItemService(
            self.repo, self.personal_vectors, self.embedder,
            cfg.embedder.personal_min_score, cfg.storage.thumb_size,
        )

        # 視覺記憶層：持續身分 + 情節/習慣
        self.identity = IdentityManager(self.repo, cfg.memory)
        self.memory = MemoryService(self.repo, cfg.memory, cfg.storage.retention_days)

        # 執行緒（Producer-Consumer）
        self.embed_queue: queue.Queue = queue.Queue(maxsize=256)
        self.writer = StorageWriter(self.repo, self.vectors, cfg.storage.retention_days,
                                    identity=self.identity)
        self.capture = CaptureService(cfg.camera, self.hub)
        self.pipeline = InferencePipeline(
            hub=self.hub,
            detector=self.detector,
            tracker=self.tracker,
            gate=MotionGate(cfg.motion),
            roi=RoiMask(cfg.roi),
            settings=self.settings,
            writer=self.writer,
            embed_queue=self.embed_queue,
            area_labels=cfg.detector.area_labels,
            embed_enabled=self.embedder.enabled,
        )
        self.embed_worker = EmbeddingWorker(
            self.embedder, self.vectors, self.embed_queue, cfg.embedder.batch_size,
            personal=self.personal, identity=self.identity, repo=self.repo,
        )

    def start(self) -> None:
        removed = self.repo.cleanup(self.cfg.storage.retention_days)
        if removed:
            self.vectors.delete(removed)
        self.identity.prune(self.cfg.storage.retention_days)  # 長時間停機後補清過期身分
        self.vocab.refresh("")
        self.detector.warmup()

        self.writer.start()
        self.capture.start()
        self.pipeline.start()
        if self.embedder.enabled:
            self.embed_worker.start()
        logger.info("CtrlF 系統就緒 | detector=%s | embedder=%s | vectors=%d 筆",
                    self.detector.info["model"],
                    "CLIP" if self.embedder.enabled else "off",
                    self.vectors.count())

    def stop(self) -> None:
        self.capture.stop()
        self.pipeline.stop()
        self.embed_worker.stop()
        self.writer.stop()
        for t in (self.capture, self.pipeline, self.embed_worker, self.writer):
            if t.is_alive():
                t.join(timeout=3.0)
        self.vectors.flush()
        self.personal_vectors.flush()
        logger.info("CtrlF 已優雅關閉")
