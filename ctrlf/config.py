"""集中式設定：config.yaml -> Pydantic 模型，所有模組只依賴這裡的型別。"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# 專案根目錄（ctrlf 套件的上一層）
ROOT_DIR = Path(__file__).resolve().parents[1]


class CameraConfig(BaseModel):
    index: int = 0
    width: int = 1280
    height: int = 720
    fourcc: str = "MJPG"
    use_dshow: bool = True  # Windows MSMF 抓圖失敗的解法


class RoiConfig(BaseModel):
    """邊界 ROI 遮罩：只在矩形內偵測，框外背景直接抹黑，從源頭過濾雜訊。"""
    enabled: bool = False
    rect: list[float] = Field(default=[0.0, 0.0, 1.0, 1.0])  # 正規化 x1, y1, x2, y2


class MotionConfig(BaseModel):
    """智慧型偵測觸發：像素差分 + 推論頻率上限 + 定時強制取樣。"""
    downscale_width: int = 320
    sensitivity: float = 0.5      # 0~1，越高越容易觸發推論
    max_infer_fps: float = 5.0    # 推論頻率硬上限（顯示串流不受此限）
    force_interval: float = 3.0   # 即使畫面靜止，至少每 N 秒強制推論一次
    warmup_frames: int = 5


class DetectorConfig(BaseModel):
    # 依序嘗試載入；預設 balanced 級（m-worldv2），離線退回本地大模型
    model_candidates: list[str] = Field(default=[
        "yolov8m-worldv2.pt",
        "yolov8x-world.pt",
        "yolov8s-worldv2.pt",
    ])
    imgsz: int = 800
    conf: float = 0.15
    iou: float = 0.45
    max_det: int = 60
    half: bool = True
    device: str = "auto"  # auto | cpu | cuda
    base_labels: list[str] = Field(default=[
        "pen", "pencil", "cell phone", "key", "keys", "bottle", "cup", "mug",
        "laptop", "keyboard", "mouse", "book", "notebook", "wallet", "glasses",
        "remote control", "headphones", "earphones", "watch", "scissors",
        "charger", "power bank", "umbrella", "backpack", "handbag",
        "tissue box", "comb", "lighter",
    ])
    area_labels: list[str] = Field(default=[
        "bed", "desk", "table", "nightstand", "shelf",
        "chair", "cabinet", "floor", "sofa", "windowsill",
    ])


class TrackerConfig(BaseModel):
    iou_threshold: float = 0.25
    center_dist_frac: float = 0.08  # 質心備援匹配距離（佔畫面對角線比例）
    confirm_hits: int = 2           # 連續命中 N 次才確認（過濾單幀誤檢）
    ttl: float = 5.0                # 失蹤超過 N 秒移除軌跡
    move_rebind_px: int = 60        # 位移超過 N px 視為「換了位置」，更新資料庫
    heartbeat: float = 10.0         # 持續可見時，至少每 N 秒刷新 last_seen


class EmbedderConfig(BaseModel):
    enabled: bool = True
    model: str = "ViT-B/32"   # OpenAI CLIP（YOLO-World 已快取同款權重，零下載）
    batch_size: int = 8
    min_score: float = 0.22   # 語意搜尋的最低餘弦相似度
    personal_min_score: float = 0.75  # 「我的物品」影像對影像比對門檻


class MemoryConfig(BaseModel):
    """視覺記憶層：身分持續性 + 情節 + 習慣。"""
    identity_threshold: float = 0.82   # 同類別下合併為同一身分的餘弦門檻
    identity_loc_boost: float = 0.04   # 一小時內同地點再現 -> 門檻放寬量（時空先驗）
    prototype_alpha: float = 0.2       # 原型 EMA 更新率
    episode_gap_min: float = 30.0      # 同地點兩次目擊間隔 < N 分鐘視為同一情節
    habit_min_minutes: float = 10.0    # 習慣統計的最低累積停留時間


class StorageConfig(BaseModel):
    data_dir: str = "data"
    retention_days: int = 7
    thumb_size: int = 160     # 歷史縮圖最長邊


class StreamConfig(BaseModel):
    jpeg_quality: int = 80
    max_fps: int = 24


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class AppConfig(BaseModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    roi: RoiConfig = Field(default_factory=RoiConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    stream: StreamConfig = Field(default_factory=StreamConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    p = Path(path) if path else ROOT_DIR / "config.yaml"
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    else:
        data = {}
    return AppConfig(**data)
