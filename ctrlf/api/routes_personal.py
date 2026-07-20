"""個人物品（照片註冊 + 註冊管理）與中文同義詞的管理路由。"""
from __future__ import annotations

import base64
import logging

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger("ctrlf.api.personal")

router = APIRouter(tags=["personal"])


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1)          # 顯示名稱，可中文：「我的黑色錢包」
    label: str = Field(min_length=1)         # 偵測類別（英文），如 wallet
    images: list[str] = Field(min_length=1)  # base64 JPEG/PNG，建議 3~5 張


class ImagesRequest(BaseModel):
    images: list[str] = Field(min_length=1)


class TestRequest(BaseModel):
    image: str = Field(min_length=1)


class RenameRequest(BaseModel):
    name: str = Field(min_length=1)


class AliasRequest(BaseModel):
    alias: str = Field(min_length=1)
    label: str = Field(min_length=1)


def _b64(blob: bytes | None) -> str | None:
    return base64.b64encode(blob).decode("ascii") if blob else None


def _decode_image(b64: str) -> np.ndarray | None:
    try:
        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img if img is not None and img.size > 0 else None
    except Exception:  # noqa: BLE001
        return None


def _decode_images(b64_list: list[str]) -> list[np.ndarray]:
    return [img for img in map(_decode_image, b64_list) if img is not None]


# ---- 我的物品 ----

@router.get("/items")
def list_items(request: Request):
    c = request.app.state.container
    items = []
    for it in c.repo.personal_items():
        out = dict(it)
        out["thumb"] = _b64(out.pop("last_thumb", None) or out.get("thumb"))
        items.append(out)
    return {"items": items, "available": c.personal.available}


@router.get("/items/{item_id}")
def item_detail(item_id: str, request: Request):
    c = request.app.state.container
    item = c.repo.get_personal_item(item_id)
    if item is None:
        raise HTTPException(404, "找不到該物品")
    out = dict(item)
    out["thumb"] = _b64(out.pop("last_thumb", None) or out.get("thumb"))
    out["photos"] = [
        {"photo_id": p["photo_id"], "thumb": _b64(p["thumb"]), "created_at": p["created_at"]}
        for p in c.repo.personal_photos(item_id)
    ]
    return out


@router.post("/items")
def register_item(req: RegisterRequest, request: Request):
    c = request.app.state.container
    if not c.personal.available:
        raise HTTPException(503, "CLIP 嵌入器未啟用，無法使用照片註冊功能")
    if not req.label.strip().isascii():
        raise HTTPException(400, "物品類別請使用英文（例如 wallet、keys）")

    images = _decode_images(req.images)
    if not images:
        raise HTTPException(400, "沒有可解析的圖片，請上傳 JPEG/PNG")

    logger.info("註冊請求: name=%s label=%s photos=%d", req.name, req.label, len(images))
    try:
        result = c.personal.register(req.name.strip(), req.label, images)
    except Exception as e:  # noqa: BLE001 — 把真實原因回給 UI 並留下完整日誌
        logger.exception("註冊處理失敗")
        raise HTTPException(500, f"註冊處理失敗：{e}") from e
    if result is None:
        raise HTTPException(500, "影像特徵編碼失敗，請換幾張更清晰的照片")
    c.vocab.refresh()  # 物品類別立即加入偵測詞彙
    return result


@router.post("/items/{item_id}/photos")
def add_photos(item_id: str, req: ImagesRequest, request: Request):
    c = request.app.state.container
    if c.repo.get_personal_item(item_id) is None:
        raise HTTPException(404, "找不到該物品")
    images = _decode_images(req.images)
    if not images:
        raise HTTPException(400, "沒有可解析的圖片")
    try:
        added = c.personal.add_photos(item_id, images)
    except Exception as e:  # noqa: BLE001
        logger.exception("補充照片失敗")
        raise HTTPException(500, f"補充照片失敗：{e}") from e
    if added == 0:
        raise HTTPException(500, "影像特徵編碼失敗")
    return {"status": "ok", "added": added}


@router.delete("/items/{item_id}/photos/{photo_id}")
def remove_photo(item_id: str, photo_id: str, request: Request):
    c = request.app.state.container
    if c.repo.get_personal_item(item_id) is None:
        raise HTTPException(404, "找不到該物品")
    if len(c.repo.personal_photos(item_id)) <= 1:
        raise HTTPException(400, "至少要保留一張參考照；要整件移除請刪除物品")
    c.personal.remove_photo(item_id, photo_id)
    return {"status": "ok"}


@router.patch("/items/{item_id}")
def rename_item(item_id: str, req: RenameRequest, request: Request):
    c = request.app.state.container
    if c.repo.get_personal_item(item_id) is None:
        raise HTTPException(404, "找不到該物品")
    c.repo.rename_personal_item(item_id, req.name)
    return {"status": "ok", "name": req.name.strip()}


@router.post("/items/{item_id}/test")
def test_item(item_id: str, req: TestRequest, request: Request):
    """註冊品質檢測：上傳一張現場照，回報能否被認出。"""
    c = request.app.state.container
    if c.repo.get_personal_item(item_id) is None:
        raise HTTPException(404, "找不到該物品")
    image = _decode_image(req.image)
    if image is None:
        raise HTTPException(400, "圖片無法解析")
    score = c.personal.test_match(item_id, image)
    if score is None:
        raise HTTPException(500, "比對失敗：CLIP 未啟用或該物品沒有參考照")
    threshold = c.cfg.embedder.personal_min_score
    return {"score": round(score, 4), "threshold": threshold,
            "passed": score >= threshold}


@router.delete("/items/{item_id}")
def delete_item(item_id: str, request: Request):
    c = request.app.state.container
    if not c.personal.delete(item_id):
        raise HTTPException(404, "找不到該物品")
    c.vocab.refresh()
    return {"status": "ok"}


# ---- 中文同義詞 ----

@router.get("/aliases")
def list_aliases(request: Request):
    c = request.app.state.container
    return {"custom": c.repo.aliases(), "builtin_count": c.lexicon.builtin_size}


@router.post("/aliases")
def add_alias(req: AliasRequest, request: Request):
    c = request.app.state.container
    if not req.label.strip().isascii():
        raise HTTPException(400, "對應標籤請使用英文（例如 wallet）")
    c.repo.add_alias(req.alias, req.label)
    c.lexicon.reload(c.repo.aliases())
    return {"status": "ok", "custom": c.repo.aliases()}


@router.delete("/aliases/{alias}")
def remove_alias(alias: str, request: Request):
    c = request.app.state.container
    c.repo.remove_alias(alias)
    c.lexicon.reload(c.repo.aliases())
    return {"status": "ok", "custom": c.repo.aliases()}
