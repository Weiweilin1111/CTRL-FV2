"""資料路由：搜尋、即時狀態、歷史紀錄、語意搜尋、標籤管理。"""
from __future__ import annotations

import base64
import difflib

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=["data"])


class SearchRequest(BaseModel):
    query: str = ""


class TagRequest(BaseModel):
    tag: str


def _b64(blob: bytes | None) -> str | None:
    return base64.b64encode(blob).decode("ascii") if blob else None


def _public(record: dict) -> dict:
    out = dict(record)
    out["thumb"] = _b64(out.get("thumb"))
    return out


@router.post("/search")
def search(req: SearchRequest, request: Request):
    """設定當前搜尋目標：辭典解析（中→英）+ 高亮 + 重建偵測詞彙 + 記錄近期標籤。"""
    c = request.app.state.container
    raw = req.query.strip()
    resolved, via = c.lexicon.resolve(raw)
    if resolved is None:  # 中文且辭典查無對應 -> 交由 UI 引導補同義詞
        return {"resolved": False, "original": raw,
                "query": c.settings.get("query"), "via": None}

    query = resolved.strip().lower()
    c.settings.update(query=query)
    labels = c.vocab.refresh(query)
    if query:
        c.repo.add_tag(query, "recent")
    return {"resolved": True, "original": raw, "query": query,
            "via": via, "classes": len(labels)}


@router.get("/state")
def state(request: Request):
    c = request.app.state.container
    snapshot = c.hub.snapshot()
    snapshot["query"] = c.settings.get("query")
    return snapshot


@router.get("/history")
def history(query: str = "", limit: int = 5, *, request: Request):
    c = request.app.state.container
    records = c.repo.history(query, limit=max(1, min(20, limit)))
    return {"records": [_public(r) for r in records]}


@router.get("/semantic")
def semantic(query: str = "", k: int = 6, *, request: Request):
    """跨模態語意搜尋：CLIP 文字向量 vs 歷史目擊的 CLIP 影像向量。

    CLIP 不可用時，優雅降級為已知標籤的模糊字串比對。
    """
    c = request.app.state.container
    query = query.strip()
    if not query:
        return {"mode": "none", "results": []}
    resolved, _ = c.lexicon.resolve(query)  # CLIP 文字端吃英文，先過辭典
    if resolved is None:
        return {"mode": "unresolved", "results": []}
    query = resolved

    text_vec = c.embedder.encode_text(query) if c.embedder.enabled else None
    if text_vec is None:
        labels = c.repo.distinct_labels()
        suggestions = difflib.get_close_matches(query.lower(), labels, n=3, cutoff=0.5)
        return {"mode": "label", "suggestions": suggestions, "results": []}

    hits = c.vectors.search(text_vec, k=max(1, min(12, k)),
                            min_score=c.cfg.embedder.min_score)
    rows = c.repo.sightings_by_ids([h["id"] for h in hits])
    results = []
    for h in hits:
        row = rows.get(h["id"])
        if row is None:  # 向量還在但 SQLite 已 TTL 清掉 -> 略過
            continue
        merged = _public(row)
        merged["score"] = h["score"]
        results.append(merged)
    return {"mode": "visual", "results": results}


@router.get("/tags")
def all_tags(request: Request):
    c = request.app.state.container
    return {"fixed": c.repo.tags("fixed"), "recent": c.repo.tags("recent")}


@router.post("/tags/fixed")
def add_fixed_tag(req: TagRequest, request: Request):
    c = request.app.state.container
    c.repo.add_tag(req.tag, "fixed")
    c.vocab.refresh()  # 常駐標籤立即進入偵測詞彙
    return {"status": "ok", "fixed": c.repo.tags("fixed")}


@router.delete("/tags/fixed/{tag}")
def remove_fixed_tag(tag: str, request: Request):
    c = request.app.state.container
    c.repo.remove_tag(tag, "fixed")
    c.vocab.refresh()
    return {"status": "ok", "fixed": c.repo.tags("fixed")}
