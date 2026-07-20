"""視覺記憶查詢路由：物件身分總覽、情節時間軸、回溯查詢。"""
from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/memory", tags=["memory"])


def _b64(blob: bytes | None) -> str | None:
    return base64.b64encode(blob).decode("ascii") if blob else None


@router.get("/objects")
def memory_objects(request: Request):
    """所有持續身分的總覽（含最後行蹤與最常出沒地）。"""
    c = request.app.state.container
    return {"objects": c.memory.objects(), "identities": c.identity.count()}


@router.get("/timeline")
def memory_timeline(object_id: str, request: Request):
    """單一物件的情節時間軸 + 習慣分佈。"""
    c = request.app.state.container
    episodes = c.memory.episodes(object_id=object_id)
    if not episodes and not any(i["object_id"] == object_id for i in c.repo.identities_meta()):
        raise HTTPException(404, "找不到該物件身分")
    return {
        "episodes": [e.to_dict() for e in episodes],
        "habits": c.memory.habits_from(episodes),
    }


@router.get("/recall")
def memory_recall(query: str = "", *, request: Request):
    """「我最後一次看到 X 是在哪？」—— 中文先過辭典，再以身分為單位回答。"""
    c = request.app.state.container
    resolved, _ = c.lexicon.resolve(query)
    if resolved is None:
        return {"label": None, "objects": []}
    results = []
    for obj in c.memory.recall(resolved):
        out = dict(obj)
        out["thumb"] = _b64(out.get("thumb"))
        results.append(out)
    return {"label": resolved, "objects": results}
