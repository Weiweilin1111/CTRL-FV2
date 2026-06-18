"""CtrlF 後端啟動入口： python run_api.py [--config config.yaml]"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)  # 模型/資料路徑以專案根為準

    parser = argparse.ArgumentParser(description="CtrlF API server")
    parser.add_argument("--config", default=None, help="config.yaml 路徑")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    from ctrlf.api.app import create_app
    from ctrlf.config import load_config

    cfg = load_config(args.config)
    import uvicorn

    uvicorn.run(
        create_app(args.config),
        host=args.host or cfg.server.host,
        port=args.port or cfg.server.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
