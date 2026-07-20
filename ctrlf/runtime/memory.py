"""MemoryService — 情節建構與習慣固化（事件 → 情節 → 習慣）。

設計取捨：情節（Episode）採「查詢時惰性建構」而非物化表——
身分指派發生在非同步嵌入執行緒，物化會引入排序競態；而 7 天 TTL
的資料量級（數千列）在查詢時合併只需毫秒級，且永遠與事實一致。

- Episode：同一身分、同一地點、相鄰目擊間隔 < gap 分鐘 -> 合併為一段情節。
- Habit：依「停留時間加權」的地點分佈（用次數加權會被閃爍目擊污染）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ctrlf.config import MemoryConfig
from ctrlf.storage.sqlite_repo import Repository

_TS_FMT = "%Y-%m-%d %H:%M:%S"


def _parse(ts: str) -> datetime:
    return datetime.strptime(ts, _TS_FMT)


@dataclass(slots=True)
class Episode:
    object_id: str
    label: str
    location: str
    start: str
    end: str

    @property
    def minutes(self) -> float:
        try:
            return max(0.0, (_parse(self.end) - _parse(self.start)).total_seconds() / 60.0)
        except ValueError:
            return 0.0

    def to_dict(self) -> dict:
        return {"object_id": self.object_id, "label": self.label,
                "location": self.location, "start": self.start, "end": self.end,
                "duration_min": round(self.minutes, 1)}


class MemoryService:
    def __init__(self, repo: Repository, cfg: MemoryConfig, retention_days: int = 7):
        self.repo = repo
        self.cfg = cfg
        self.retention_days = retention_days

    # ---- Episode：事件 -> 情節 ----

    def episodes(self, object_id: str | None = None,
                 label_like: str | None = None) -> list[Episode]:
        rows = self.repo.memory_rows(self.retention_days, object_id, label_like)
        episodes: list[Episode] = []
        open_eps: dict[str, Episode] = {}  # object_id -> 進行中情節

        for row in rows:
            oid, loc = row["object_id"], row["location"] or "unknown"
            current = open_eps.get(oid)
            if current is not None and current.location == loc:
                try:
                    gap_min = (_parse(row["first_seen"]) - _parse(current.end)).total_seconds() / 60.0
                except ValueError:
                    gap_min = float("inf")
                if gap_min < self.cfg.episode_gap_min:
                    if row["last_seen"] > current.end:
                        current.end = row["last_seen"]
                    continue
            if current is not None:
                episodes.append(current)
            open_eps[oid] = Episode(oid, row["label"], loc,
                                    row["first_seen"], row["last_seen"])
        episodes.extend(open_eps.values())
        episodes.sort(key=lambda e: e.start, reverse=True)
        return episodes

    # ---- Habit：情節 -> 習慣（語意記憶） ----

    def habits(self, object_id: str) -> list[dict]:
        return self.habits_from(self.episodes(object_id=object_id))

    def habits_from(self, episodes: list[Episode]) -> list[dict]:
        """由現成的情節清單計算習慣——熱路徑已算過 episodes 時不必重查一遍。"""
        minutes_by_loc: dict[str, float] = {}
        for ep in episodes:
            minutes_by_loc[ep.location] = minutes_by_loc.get(ep.location, 0.0) + max(ep.minutes, 0.5)
        total = sum(minutes_by_loc.values())
        if total < self.cfg.habit_min_minutes:
            return []  # 觀測量不足，不下結論
        ranked = sorted(minutes_by_loc.items(), key=lambda kv: -kv[1])
        return [{"location": loc, "pct": round(mins / total, 3), "minutes": round(mins, 1)}
                for loc, mins in ranked]

    # ---- 物件總覽與回溯查詢 ----

    def objects(self) -> list[dict]:
        # 一次查詢建出所有物件的情節再分組——身分數成長時仍維持 2 次查詢
        eps_by_oid: dict[str, list[Episode]] = {}
        for ep in self.episodes():
            eps_by_oid.setdefault(ep.object_id, []).append(ep)
        out = []
        for ident in self.repo.identities_meta():
            habits = self.habits_from(eps_by_oid.get(ident["object_id"], []))
            out.append({
                "object_id": ident["object_id"],
                "display": ident["display"],
                "label": ident["label"],
                "n_obs": ident["n_obs"],
                "first_seen": ident["first_seen"],
                "last_seen": ident["last_seen"],
                "last_location": ident["last_location"],
                "top_habit": habits[0] if habits else None,
            })
        return out

    def recall(self, label: str, limit: int = 3) -> list[dict]:
        """「我最後一次看到 X 是在哪？」—— 以身分為單位回答。

        這是 UI 每 1.5 秒輪詢的熱路徑：身分清單已依 last_seen 排序，
        先過濾類別取前 limit 個，只對這幾件計算情節/習慣/縮圖。
        """
        label = (label or "").strip().lower()
        matched = []
        for ident in self.repo.identities_meta():  # 已依 last_seen DESC 排序
            if label and label not in ident["label"] and ident["label"] not in label:
                continue
            matched.append(ident)
            if len(matched) >= limit:
                break

        results = []
        for ident in matched:
            oid = ident["object_id"]
            eps = self.episodes(object_id=oid)
            habits = self.habits_from(eps)
            results.append({
                "object_id": oid,
                "display": ident["display"],
                "label": ident["label"],
                "n_obs": ident["n_obs"],
                "last_seen": ident["last_seen"],
                "last_location": ident["last_location"],
                "top_habit": habits[0] if habits else None,
                "latest_episode": eps[0].to_dict() if eps else None,
                "thumb": self.repo.latest_thumb_for_object(oid),
            })
        return results
