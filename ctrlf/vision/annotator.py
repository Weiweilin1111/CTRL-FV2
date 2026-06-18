"""Annotator — 純顯示層：把追蹤結果畫到影格上（與推論完全解耦）。

在串流端以「最新影格 + 最新軌跡」即時合成，推論慢不影響畫面流暢度。
命中目標 = 半透明填色 + 粗框 + 角標 + 標籤晶片；其餘物件可選淡色細框。
"""
from __future__ import annotations

import time

import cv2
import numpy as np

HIGHLIGHT = (90, 255, 120)   # BGR 螢光綠
NEUTRAL = (160, 140, 110)    # 淡藍灰
TEXT_DARK = (20, 30, 20)


def _query_match(query: str, label: str) -> bool:
    return bool(query) and (query in label or label in query)


class Annotator:
    def draw(self, frame: np.ndarray, tracks: list[dict], query: str, show_all: bool) -> np.ndarray:
        out = frame.copy()
        q = (query or "").strip().lower()
        matched_any = False

        for t in tracks:
            match = _query_match(q, t["label"])
            if not match and not show_all:
                continue
            x1, y1, x2, y2 = t["bbox"]
            if match:
                matched_any = True
                self._draw_highlight(out, x1, y1, x2, y2)
                self._chip(out, x1, y1, f"{t['display']}  {t['confidence']:.2f}", HIGHLIGHT)
            else:
                cv2.rectangle(out, (x1, y1), (x2, y2), NEUTRAL, 1)
                cv2.putText(out, t["display"], (x1, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, NEUTRAL, 1, cv2.LINE_AA)

        if q and not matched_any and q.isascii():
            self._banner(out, f"Scanning for '{q}' ...")

        ts = time.strftime("%H:%M:%S")
        cv2.putText(out, ts, (out.shape[1] - 90, out.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        return out

    @staticmethod
    def _draw_highlight(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
        sub = img[y1:y2, x1:x2]
        if sub.size > 0:  # 15% 半透明填色凸顯目標
            tint = np.full_like(sub, HIGHLIGHT)
            cv2.addWeighted(sub, 0.85, tint, 0.15, 0, dst=sub)
        cv2.rectangle(img, (x1, y1), (x2, y2), HIGHLIGHT, 2)
        tick = max(10, min(18, (x2 - x1) // 6))
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(img, (cx, cy), (cx + dx * tick, cy), HIGHLIGHT, 4)
            cv2.line(img, (cx, cy), (cx, cy + dy * tick), HIGHLIGHT, 4)

    @staticmethod
    def _chip(img: np.ndarray, x: int, y: int, text: str, color: tuple) -> None:
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(th + 8, y - 6)
        cv2.rectangle(img, (x, ty - th - 6), (x + tw + 10, ty + baseline - 2), color, -1)
        cv2.putText(img, text, (x + 5, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    TEXT_DARK, 1, cv2.LINE_AA)

    @staticmethod
    def _banner(img: np.ndarray, text: str) -> None:
        h = img.shape[0]
        cv2.rectangle(img, (12, h - 44), (24 + 9 * len(text), h - 16), (30, 30, 30), -1)
        cv2.putText(img, text, (20, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (120, 255, 150), 1, cv2.LINE_AA)
