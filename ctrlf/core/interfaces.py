"""抽象介面（Protocol）：依賴注入的契約層。

容器只組裝符合這些 Protocol 的實作；要換掉任何一塊
（例如把 NumpyVectorStore 換成 Qdrant、把 YOLO-World 換成其他開放詞彙模型），
只需提供同介面的新類別，不必動其他模組。
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .models import Detection


@runtime_checkable
class Detector(Protocol):
    def set_classes(self, labels: Sequence[str]) -> None: ...
    def detect(self, frame: np.ndarray, conf: float) -> list[Detection]: ...
    def warmup(self) -> None: ...
    @property
    def info(self) -> dict: ...


@runtime_checkable
class Embedder(Protocol):
    enabled: bool
    def encode_images(self, crops: Sequence[np.ndarray]) -> np.ndarray | None: ...
    def encode_text(self, text: str) -> np.ndarray | None: ...


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, vid: str, vector: np.ndarray, payload: dict) -> None: ...
    def search(self, vector: np.ndarray, k: int = 6, min_score: float = 0.0) -> list[dict]: ...
    def delete(self, ids: Sequence[str]) -> None: ...
    def flush(self) -> None: ...
    def count(self) -> int: ...
