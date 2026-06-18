"""CtrlF — 即時智慧尋物系統（v3 全面重構版）。

四大模塊完全解耦：
- capture  資料採集（相機 + ROI 遮罩 + 動態閘控）
- vision   AI 核心（YOLO-World 偵測 + IoU 追蹤 + CLIP 視覺特徵）
- storage  特徵儲存（SQLite WAL + 內嵌向量索引）
- api/ui   前端展示（FastAPI 非同步 + Streamlit）
"""
__version__ = "3.1.0"
