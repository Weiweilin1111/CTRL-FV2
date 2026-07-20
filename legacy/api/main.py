#cspell:ignore FOURCC MJPG imencode tobytes asyncio fastapi pydantic cv2
import cv2
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import sys
import os
# 將上一層目錄加入 sys.path 以便 import vision.detector
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.detector import ObjectDetector
from api.database import init_db

app = FastAPI(title="CtrlF V2 API")

@app.on_event("startup")
def startup_event():
    init_db()
    from api.database import get_fixed_tags
    fixed = get_fixed_tags()
    detector.set_targets([], background_targets=fixed)

# 初始化 ObjectDetector 與相機
detector = ObjectDetector()
# 加入 cv2.CAP_DSHOW 參數解決 Windows MSMF 抓圖失敗 (can't grab frame. Error: -1072875772) 的問題
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

class SearchQuery(BaseModel):
    query: str

async def generate_frames():
    """
    抓取影像、推論並轉換為 streaming 格式。
    使用 asyncio 確保不阻塞 FastAPI 的事件迴圈。
    """
    loop = asyncio.get_event_loop()
    while True:
        # 使用 run_in_executor 來避免 OpenCV 和推論阻塞 async 迴圈
        success, frame = await loop.run_in_executor(None, camera.read)
        if not success:
            await asyncio.sleep(0.1)
            continue
        
        # 進行推論並畫框
        annotated_frame, detections = await loop.run_in_executor(None, detector.process, frame)
        
        # 將圖片編碼成 JPEG 格式
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        # 產生 Content-Type multipart/x-mixed-replace 格式
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # 強迫讓出執行權給其他非同步任務
        await asyncio.sleep(0.01)

@app.get("/video_feed")
async def video_feed():
    """
    影像串流端點
    """
    return StreamingResponse(
        generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/search")
async def search(target: SearchQuery):
    """
    更新搜尋目標
    """
    from api.database import get_fixed_tags
    fixed = get_fixed_tags()
    targets = [target.query] if target.query else []
    detector.set_targets(targets, background_targets=fixed)
    return {"status": "success", "targets_set": targets}

@app.get("/search_get")  # 提供 GET 方法也可以方便測試
async def search_get(query: str = ""):
    from api.database import get_fixed_tags
    fixed = get_fixed_tags()
    targets = [query] if query else []
    detector.set_targets(targets, background_targets=fixed)
    return {"status": "success", "targets_set": targets}

@app.on_event("shutdown")
def release_camera():
    if camera.isOpened():
        camera.release()

# ==========================================
# 前後端解耦 API 端點 (UI 直接呼叫以下 API)
# ==========================================
from api.database import (
    get_history, add_recent_tag, get_recent_tags, 
    get_fixed_tags, add_fixed_tag, remove_fixed_tag, get_semantic_closest
)

class TagRequest(BaseModel):
    tag: str

@app.get("/history")
async def api_get_history(query: str):
    records = get_history(query)
    return {"records": records}

@app.get("/semantic_closest")
async def api_get_semantic_closest(query: str):
    closest_tag, dist = get_semantic_closest(query)
    return {"closest_tag": closest_tag, "distance": dist}

@app.get("/recent_tags")
async def api_get_recent_tags():
    tags = get_recent_tags()
    return {"tags": tags}

@app.post("/recent_tags")
async def api_add_recent_tag(req: TagRequest):
    add_recent_tag(req.tag)
    return {"status": "success"}

@app.get("/fixed_tags")
async def api_get_fixed_tags():
    tags = get_fixed_tags()
    return {"tags": tags}

@app.post("/fixed_tags")
async def api_add_fixed_tag(req: TagRequest):
    add_fixed_tag(req.tag)
    return {"status": "success"}

@app.delete("/fixed_tags/{tag}")
async def api_remove_fixed_tag(tag: str):
    remove_fixed_tag(tag)
    return {"status": "success"}
