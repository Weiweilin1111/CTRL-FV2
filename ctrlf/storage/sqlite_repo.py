"""Repository — SQLite 持久層（WAL 模式，讀寫不互鎖）。

Schema 改造重點：
- sightings 以「軌跡生命週期」為一列（sighting_id = 追蹤器確認時配發的 UUID），
  天然去重，不再依賴舊版 50px/10min 的啟發式判斷。
- 內建縮圖 BLOB：歷史紀錄直接顯示「上次看到它長什麼樣、在哪」。
- fixed/recent 兩張表合併為 tags(kind)。
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from ctrlf.core.models import SightingEvent

logger = logging.getLogger("ctrlf.repo")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sightings (
    sighting_id TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    instance    INTEGER NOT NULL DEFAULT 0,
    location    TEXT,
    bbox        TEXT,
    confidence  REAL,
    first_seen  TEXT,
    last_seen   TEXT,
    thumb       BLOB
);
CREATE INDEX IF NOT EXISTS idx_sightings_label ON sightings(label, last_seen DESC);
CREATE INDEX IF NOT EXISTS idx_sightings_seen  ON sightings(last_seen);
CREATE TABLE IF NOT EXISTS tags (
    tag       TEXT NOT NULL,
    kind      TEXT NOT NULL CHECK (kind IN ('fixed', 'recent')),
    last_used TEXT,
    PRIMARY KEY (tag, kind)
);
CREATE TABLE IF NOT EXISTS aliases (
    alias TEXT PRIMARY KEY,
    label TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS personal_items (
    item_id       TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    label         TEXT NOT NULL,
    n_photos      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT,
    thumb         BLOB,
    last_seen     TEXT,
    last_location TEXT,
    last_score    REAL,
    last_thumb    BLOB
);
CREATE TABLE IF NOT EXISTS personal_photos (
    photo_id   TEXT PRIMARY KEY,
    item_id    TEXT NOT NULL,
    thumb      BLOB,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_personal_photos_item ON personal_photos(item_id);
"""


class Repository:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
        logger.info("SQLite 已就緒: %s (WAL)", db_path)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- sightings ----

    def upsert_sightings(self, events: list[SightingEvent]) -> None:
        with self._conn() as conn:
            conn.executemany(
                """
                INSERT INTO sightings
                    (sighting_id, label, instance, location, bbox, confidence,
                     first_seen, last_seen, thumb)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sighting_id) DO UPDATE SET
                    location   = excluded.location,
                    bbox       = excluded.bbox,
                    confidence = excluded.confidence,
                    last_seen  = excluded.last_seen,
                    thumb      = COALESCE(excluded.thumb, sightings.thumb)
                """,
                [
                    (
                        ev.sighting_id, ev.label, ev.instance, ev.location,
                        str(list(ev.bbox)), ev.confidence,
                        ev.first_seen, ev.last_seen, ev.thumb_jpeg,
                    )
                    for ev in events
                ],
            )

    def history(self, query: str, limit: int = 5) -> list[dict]:
        like = f"%{(query or '').strip().lower()}%"
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT sighting_id, label, instance, location, confidence,
                       first_seen, last_seen, thumb
                FROM sightings WHERE label LIKE ?
                ORDER BY last_seen DESC LIMIT ?
                """,
                (like, limit),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def sightings_by_ids(self, ids: list[str]) -> dict[str, dict]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT sighting_id, label, instance, location, confidence,
                       first_seen, last_seen, thumb
                FROM sightings WHERE sighting_id IN ({marks})
                """,
                ids,
            ).fetchall()
        return {r[0]: self._row_to_dict(r) for r in rows}

    def distinct_labels(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT DISTINCT label FROM sightings").fetchall()
        return [r[0] for r in rows]

    def cleanup(self, days: int) -> list[str]:
        """TTL 清理；回傳被刪除的 sighting_id 以便同步清向量索引。"""
        limit = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT sighting_id FROM sightings WHERE last_seen < ?", (limit,)
            ).fetchall()
            ids = [r[0] for r in rows]
            if ids:
                conn.execute("DELETE FROM sightings WHERE last_seen < ?", (limit,))
                logger.info("TTL 清理 %d 筆超過 %d 天的目擊紀錄", len(ids), days)
        return ids

    @staticmethod
    def _row_to_dict(r) -> dict:
        return {
            "sighting_id": r[0],
            "label": r[1],
            "instance": r[2],
            "display": f"{r[1]}-{r[2]}" if r[2] else r[1],
            "location": r[3],
            "confidence": r[4],
            "first_seen": r[5],
            "last_seen": r[6],
            "thumb": r[7],  # bytes | None，API 層轉 base64
        }

    # ---- tags ----

    def add_tag(self, tag: str, kind: str) -> None:
        tag = (tag or "").strip().lower()
        if not tag:
            return
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO tags (tag, kind, last_used) VALUES (?, ?, ?)
                ON CONFLICT(tag, kind) DO UPDATE SET last_used = excluded.last_used
                """,
                (tag, kind, now),
            )
            if kind == "recent":  # 近期清單上限 10 筆
                conn.execute(
                    """
                    DELETE FROM tags WHERE kind = 'recent' AND tag NOT IN (
                        SELECT tag FROM tags WHERE kind = 'recent'
                        ORDER BY last_used DESC LIMIT 10
                    )
                    """
                )

    def remove_tag(self, tag: str, kind: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM tags WHERE tag = ? AND kind = ?",
                         ((tag or "").strip().lower(), kind))

    def tags(self, kind: str) -> list[str]:
        order = "last_used DESC" if kind == "recent" else "tag ASC"
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT tag FROM tags WHERE kind = ? ORDER BY {order}", (kind,)
            ).fetchall()
        return [r[0] for r in rows]

    # ---- 中文同義詞 ----

    def aliases(self) -> dict[str, str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT alias, label FROM aliases ORDER BY alias").fetchall()
        return {r[0]: r[1] for r in rows}

    def add_alias(self, alias: str, label: str) -> None:
        alias = (alias or "").strip().lower()
        label = (label or "").strip().lower()
        if not alias or not label:
            return
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO aliases (alias, label) VALUES (?, ?)
                ON CONFLICT(alias) DO UPDATE SET label = excluded.label
                """,
                (alias, label),
            )

    def remove_alias(self, alias: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM aliases WHERE alias = ?",
                         ((alias or "").strip().lower(),))

    # ---- 個人物品（照片註冊） ----

    def add_personal_item(self, item_id: str, name: str, label: str,
                          n_photos: int, thumb: bytes | None) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO personal_items (item_id, name, label, n_photos, created_at, thumb)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (item_id, name.strip(), label.strip().lower(), n_photos, now, thumb),
            )

    def personal_items(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT item_id, name, label, n_photos, created_at, thumb,
                       last_seen, last_location, last_score, last_thumb
                FROM personal_items ORDER BY created_at DESC
                """
            ).fetchall()
        return [
            {
                "item_id": r[0], "name": r[1], "label": r[2], "n_photos": r[3],
                "created_at": r[4], "thumb": r[5], "last_seen": r[6],
                "last_location": r[7], "last_score": r[8], "last_thumb": r[9],
            }
            for r in rows
        ]

    def personal_labels(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT DISTINCT label FROM personal_items").fetchall()
        return [r[0] for r in rows]

    def get_personal_item(self, item_id: str) -> dict | None:
        for item in self.personal_items():
            if item["item_id"] == item_id:
                return item
        return None

    def rename_personal_item(self, item_id: str, name: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE personal_items SET name = ? WHERE item_id = ?",
                         (name.strip(), item_id))

    def delete_personal_item(self, item_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM personal_photos WHERE item_id = ?", (item_id,))
            conn.execute("DELETE FROM personal_items WHERE item_id = ?", (item_id,))

    def add_personal_photos(self, item_id: str,
                            photos: list[tuple[str, bytes | None]]) -> None:
        """photos: [(photo_id, thumb_jpeg), ...]"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO personal_photos (photo_id, item_id, thumb, created_at) "
                "VALUES (?, ?, ?, ?)",
                [(pid, item_id, thumb, now) for pid, thumb in photos],
            )
            self._sync_photo_count(conn, item_id)

    def personal_photos(self, item_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT photo_id, thumb, created_at FROM personal_photos "
                "WHERE item_id = ? ORDER BY created_at",
                (item_id,),
            ).fetchall()
        return [{"photo_id": r[0], "thumb": r[1], "created_at": r[2]} for r in rows]

    def delete_personal_photo(self, item_id: str, photo_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM personal_photos WHERE photo_id = ? AND item_id = ?",
                         (photo_id, item_id))
            self._sync_photo_count(conn, item_id)

    @staticmethod
    def _sync_photo_count(conn, item_id: str) -> None:
        conn.execute(
            "UPDATE personal_items SET n_photos = "
            "(SELECT COUNT(*) FROM personal_photos WHERE item_id = ?) WHERE item_id = ?",
            (item_id, item_id),
        )

    def update_personal_sighting(self, item_id: str, location: str, seen_at: str,
                                 score: float, thumb: bytes | None) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE personal_items SET
                    last_seen = ?, last_location = ?, last_score = ?,
                    last_thumb = COALESCE(?, last_thumb)
                WHERE item_id = ?
                """,
                (seen_at, location, round(float(score), 4), thumb, item_id),
            )
