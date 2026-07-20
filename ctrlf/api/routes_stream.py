"""串流路由：MJPEG 影像 + WebSocket 即時狀態。

關鍵修正（對比舊版）：
- 串流只讀 StateHub 快照，不再在串流迴圈內推論 ——
  舊版每多開一個分頁就多跑一份 YOLO，是 VRAM 爆炸的元兇之一。
- 每次迭代檢查 is_disconnected()，殭屍連線即時回收。
- JPEG 編碼集中在 hub 快取，多客戶端共用同一份編碼結果。
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

router = APIRouter(tags=["stream"])


@router.get("/stream/mjpeg")
async def mjpeg_stream(request: Request):
    container = request.app.state.container
    interval = 1.0 / max(1, container.cfg.stream.max_fps)
    quality = container.cfg.stream.jpeg_quality
    loop = asyncio.get_running_loop()

    async def generate():
        while True:
            if await request.is_disconnected():
                break
            s = container.settings.snapshot()

            def draw(frame, tracks, _s=s):
                return container.annotator.draw(frame, tracks, _s["query"], _s["show_all"])

            jpeg = await loop.run_in_executor(
                None, container.hub.render_jpeg, draw, s["version"], quality
            )
            if jpeg is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            await asyncio.sleep(interval)

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.websocket("/ws/state")
async def ws_state(websocket: WebSocket):
    await websocket.accept()
    container = websocket.app.state.container
    try:
        while True:
            snapshot = container.hub.snapshot()
            snapshot["query"] = container.settings.get("query")
            await websocket.send_json(snapshot)
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError):
        pass
