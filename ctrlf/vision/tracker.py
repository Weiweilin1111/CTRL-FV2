"""IouTracker — 以 IoU 為主、質心為輔的輕量多目標追蹤器。

與舊版質心追蹤的差異：
- IoU 貪婪匹配（按信心值排序），質心距離只作備援 -> 雜亂場景不易錯接。
- 軌跡有生命週期：tentative -> confirmed（連續命中過濾單幀誤檢）-> 逾時移除。
- 逾時用「牆鐘秒數」而非幀數（舊版 1800 幀在低 FPS 下等於永不過期）。
- 追蹤器不碰資料庫：只產生 SightingEvent / EmbedJob，由下游各自消費。
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass

import cv2
import numpy as np

from ctrlf.config import TrackerConfig
from ctrlf.core.models import Detection, EmbedJob, SightingEvent


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _center(bbox: tuple) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _fmt(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


@dataclass(slots=True)
class _Track:
    track_id: int
    label: str
    bbox: tuple
    confidence: float
    max_conf: float
    hits: int
    first_seen: float
    last_seen: float
    location: str = "unknown"
    confirmed: bool = False
    sighting_id: str = ""
    instance: int = 0
    emit_bbox: tuple = (0, 0, 0, 0)
    emit_location: str = ""
    last_emit: float = 0.0


class IouTracker:
    def __init__(self, cfg: TrackerConfig, thumb_size: int = 160):
        self.cfg = cfg
        self.thumb_size = thumb_size
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1
        self._instance_counters: dict[str, int] = {}

    def update(
        self,
        detections: list[Detection],
        areas: list[Detection],
        frame: np.ndarray,
        now: float,
    ) -> tuple[list[dict], list[SightingEvent], list[EmbedJob]]:
        events: list[SightingEvent] = []
        embed_jobs: list[EmbedJob] = []
        h, w = frame.shape[:2]
        diag = math.hypot(w, h)
        unmatched = set(self._tracks.keys())

        for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
            best_id = None
            best_iou = self.cfg.iou_threshold
            for tid in unmatched:
                tr = self._tracks[tid]
                if tr.label != det.label:
                    continue
                score = _iou(tr.bbox, det.bbox)
                if score >= best_iou:
                    best_iou, best_id = score, tid
            if best_id is None:  # 質心備援（物體快速移動時 IoU 可能為 0）
                best_dist = self.cfg.center_dist_frac * diag
                cx, cy = det.center
                for tid in unmatched:
                    tr = self._tracks[tid]
                    if tr.label != det.label:
                        continue
                    tx, ty = _center(tr.bbox)
                    dist = math.hypot(cx - tx, cy - ty)
                    if dist < best_dist:
                        best_dist, best_id = dist, tid

            if best_id is not None:
                unmatched.discard(best_id)
                self._touch(self._tracks[best_id], det, areas, frame, now, events, embed_jobs)
            else:
                self._spawn(det, areas, frame, now, events, embed_jobs)

        # 逾時軌跡移除（牆鐘時間）
        for tid in list(unmatched):
            if now - self._tracks[tid].last_seen > self.cfg.ttl:
                del self._tracks[tid]

        snapshots = [self._view(t, now) for t in self._tracks.values() if t.confirmed]
        return snapshots, events, embed_jobs

    # ---- 內部邏輯 ----

    def _spawn(self, det: Detection, areas, frame, now, events, embed_jobs) -> None:
        track = _Track(
            track_id=self._next_track_id, label=det.label, bbox=det.bbox,
            confidence=det.confidence, max_conf=det.confidence, hits=1,
            first_seen=now, last_seen=now, location=self._locate(det, areas) or "unknown",
        )
        self._next_track_id += 1
        self._tracks[track.track_id] = track
        if self.cfg.confirm_hits <= 1:
            self._confirm(track, frame, now, events, embed_jobs)

    def _touch(self, track: _Track, det: Detection, areas, frame, now, events, embed_jobs) -> None:
        track.bbox = det.bbox
        track.confidence = det.confidence
        track.hits += 1
        track.last_seen = now
        location = self._locate(det, areas)
        if location:
            track.location = location

        if not track.confirmed:
            if track.hits >= self.cfg.confirm_hits:
                self._confirm(track, frame, now, events, embed_jobs)
            return

        moved = math.hypot(*(np.subtract(_center(track.bbox), _center(track.emit_bbox))))
        relocated = track.location != track.emit_location
        heartbeat = (now - track.last_emit) >= self.cfg.heartbeat
        better = det.confidence > track.max_conf + 0.05
        track.max_conf = max(track.max_conf, det.confidence)

        if moved > self.cfg.move_rebind_px or relocated or heartbeat or better:
            # 大幅移動或換區域時刷新縮圖，讓歷史紀錄顯示最新樣貌
            thumb = self._thumb(frame, track.bbox) if (moved > self.cfg.move_rebind_px or relocated) else None
            events.append(self._event(track, thumb))
            track.emit_bbox = track.bbox
            track.emit_location = track.location
            track.last_emit = now

    def _confirm(self, track: _Track, frame, now, events, embed_jobs) -> None:
        track.confirmed = True
        track.sighting_id = uuid.uuid4().hex
        instance = self._instance_counters.get(track.label, 0) + 1
        self._instance_counters[track.label] = instance
        track.instance = instance
        track.emit_bbox = track.bbox
        track.emit_location = track.location
        track.last_emit = now

        thumb = self._thumb(frame, track.bbox)
        events.append(self._event(track, thumb))

        crop = self._crop(frame, track.bbox)
        if crop is not None:
            embed_jobs.append(EmbedJob(
                sighting_id=track.sighting_id,
                payload={
                    "label": track.label,
                    "display": f"{track.label}-{track.instance}",
                    "location": track.location,
                    "last_seen": _fmt(now),
                },
                crop=crop,
            ))

    def _event(self, track: _Track, thumb: bytes | None) -> SightingEvent:
        return SightingEvent(
            sighting_id=track.sighting_id,
            label=track.label,
            instance=track.instance,
            location=track.location,
            bbox=tuple(int(v) for v in track.bbox),
            confidence=round(track.confidence, 4),
            first_seen=_fmt(track.first_seen),
            last_seen=_fmt(track.last_seen),
            thumb_jpeg=thumb,
        )

    @staticmethod
    def _locate(det: Detection, areas: list[Detection]) -> str | None:
        """物件中心落在哪個區域錨點內（取最小者 = 語意最精準的容器）。"""
        cx, cy = det.center
        best_name, best_area = None, float("inf")
        for anchor in areas:
            ax1, ay1, ax2, ay2 = anchor.bbox
            if ax1 <= cx <= ax2 and ay1 <= cy <= ay2 and anchor.area < best_area:
                best_name, best_area = anchor.label, anchor.area
        return best_name

    def _crop(self, frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        mx = int((x2 - x1) * 0.06)
        my = int((y2 - y1) * 0.06)
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(w, x2 + mx), min(h, y2 + my)
        crop = frame[y1:y2, x1:x2]
        return crop.copy() if crop.size > 0 else None

    def _thumb(self, frame: np.ndarray, bbox: tuple) -> bytes | None:
        crop = self._crop(frame, bbox)
        if crop is None:
            return None
        ch, cw = crop.shape[:2]
        scale = self.thumb_size / max(ch, cw)
        if scale < 1.0:
            crop = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))))
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return buf.tobytes() if ok else None

    @staticmethod
    def _view(track: _Track, now: float) -> dict:
        return {
            "track_id": track.track_id,
            "sighting_id": track.sighting_id,
            "label": track.label,
            "instance": track.instance,
            "display": f"{track.label}-{track.instance}",
            "bbox": [int(v) for v in track.bbox],
            "confidence": round(track.confidence, 3),
            "location": track.location,
            "age": round(now - track.first_seen, 1),
        }
