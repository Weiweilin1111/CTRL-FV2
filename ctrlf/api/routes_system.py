"""系統路由：效能監控與執行期參數熱調整（UI 滑桿的後端）。"""
from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/system", tags=["system"])


class SettingsPatch(BaseModel):
    conf: float | None = Field(default=None, ge=0.01, le=0.9)
    motion_sensitivity: float | None = Field(default=None, ge=0.0, le=1.0)
    max_infer_fps: float | None = Field(default=None, ge=0.5, le=15.0)
    show_all: bool | None = None


@router.get("/stats")
def stats(request: Request):
    import ctrlf
    c = request.app.state.container
    snapshot = c.hub.snapshot()
    payload = {
        "version": ctrlf.__version__,
        **snapshot["stats"],
        "tracked": len(snapshot["tracks"]),
        "detector": c.detector.info,
        "embedder": "clip" if c.embedder.enabled else "off",
        "vectors": c.vectors.count(),
    }
    try:
        from ctrlf.config import ROOT_DIR
        data_dir = ROOT_DIR / c.cfg.storage.data_dir
        size = sum(f.stat().st_size for f in data_dir.glob("*") if f.is_file())
        payload["storage_mb"] = round(size / 2**20, 2)
        payload["retention_days"] = c.cfg.storage.retention_days
    except OSError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            payload["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "vram_used_mb": round(torch.cuda.memory_reserved(0) / 2**20),
                "vram_total_mb": round(
                    torch.cuda.get_device_properties(0).total_memory / 2**20
                ),
            }
    except Exception:  # noqa: BLE001
        pass
    return payload


@router.get("/settings")
def get_settings(request: Request):
    return request.app.state.container.settings.snapshot()


@router.post("/settings")
def patch_settings(patch: SettingsPatch, request: Request):
    c = request.app.state.container
    return c.settings.update(**patch.model_dump(exclude_none=True))
