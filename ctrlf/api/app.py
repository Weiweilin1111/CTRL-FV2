"""FastAPI 應用工廠：lifespan 管理容器生命週期（取代已棄用的 on_event）。"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import ctrlf
from ctrlf.api import routes_data, routes_personal, routes_stream, routes_system
from ctrlf.config import load_config
from ctrlf.container import AppContainer


def create_app(config_path: str | None = None) -> FastAPI:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = load_config(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = AppContainer(cfg)  # 模型載入等重活集中在啟動期
        container.start()
        app.state.container = container
        yield
        container.stop()

    app = FastAPI(title="CtrlF API", version=ctrlf.__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Streamlit iframe 與本機開發
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(routes_stream.router)
    app.include_router(routes_data.router)
    app.include_router(routes_personal.router)
    app.include_router(routes_system.router)
    return app
