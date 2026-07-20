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
import threading
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
    SCHEMA_VERSION = 1

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()  # 每執行緒一條長連線，避免高頻操作反覆 connect
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)
        logger.info("SQLite 已就緒: %s (WAL, schema v%d)", db_path, self.SCHEMA_VERSION)

    def _migrate(self, conn) -> None:
        """以 PRAGMA user_version 管理 schema 演進，保護既有資料。"""
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version < 1:
            # v1: 視覺記憶層 —— sightings 掛上持續身分、新增 object_identities
            cols = [r[1] for r in conn.execute("PRAGMA table_info(sightings)").fetchall()]
            if "object_id" not in cols:
                conn.execute("ALTER TABLE sightings ADD COLUMN object_id TEXT")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS object_identities (
                    object_id     TEXT PRIMARY KEY,
                    label         TEXT NOT NULL,
                    display       TEXT NOT NULL,
                    prototype     BLOB,
                    n_obs         INTEGER NOT NULL DEFAULT 0,
                    first_seen    TEXT,
                    last_seen     TEXT,
                    last_location TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_sightings_object ON sightings(object_id, last_seen);
            """)
            conn.execute("PRAGMA user_version = 1")
            logger.info("Schema 遷移 v0 -> v1（視覺記憶層）完成")

    @contextmanager
    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()  # 連線可重用，不可留下未結束的交易
            raise

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

    def sightings_with_thumbs(self) -> list[dict]:
        """向量索引重建的原料：所有帶縮圖的目擊列（tools/rebuild_vectors.py 用）。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT sighting_id, label, instance, location, last_seen, thumb "
                "FROM sightings WHERE thumb IS NOT NULL"
            ).fetchall()
        return [
            {"sighting_id": r[0], "label": r[1], "instance": r[2],
             "location": r[3], "last_seen": r[4], "thumb": r[5]}
            for r in rows
        ]

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

    # ---- 視覺記憶層（身分 / 情節原料） ----

    def upsert_identity(self, object_id: str, label: str, display: str,
                        prototype: bytes, n_obs: int,
                        first_seen: str, last_seen: str, last_location: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO object_identities
                    (object_id, label, display, prototype, n_obs,
                     first_seen, last_seen, last_location)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(object_id) DO UPDATE SET
                    prototype = excluded.prototype,
                    n_obs = excluded.n_obs,
                    last_seen = excluded.last_seen,
                    last_location = excluded.last_location
                """,
                (object_id, label, display, prototype, n_obs,
                 first_seen, last_seen, last_location),
            )

    def identities(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT object_id, label, display, prototype, n_obs,
                       first_seen, last_seen, last_location
                FROM object_identities ORDER BY last_seen DESC
                """
            ).fetchall()
        return [
            {"object_id": r[0], "label": r[1], "display": r[2], "prototype": r[3],
             "n_obs": r[4], "first_seen": r[5], "last_seen": r[6], "last_location": r[7]}
            for r in rows
        ]

    def identities_meta(self) -> list[dict]:
        """身分清單（不含 prototype BLOB）——查詢熱路徑用，省去每列 2KB 的搬運。"""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT object_id, label, display, n_obs,
                       first_seen, last_seen, last_location
                FROM object_identities ORDER BY last_seen DESC
                """
            ).fetchall()
        return [
            {"object_id": r[0], "label": r[1], "display": r[2], "n_obs": r[3],
             "first_seen": r[4], "last_seen": r[5], "last_location": r[6]}
            for r in rows
        ]

    def delete_identities(self, ids: list[str]) -> None:
        if not ids:
            return
        with self._conn() as conn:
            conn.executemany("DELETE FROM object_identities WHERE object_id = ?",
                             [(oid,) for oid in ids])

    def stale_identity_ids(self, days: int) -> list[str]:
        """超過保留期未活躍的身分。

        「活躍」取身分列 last_seen 與其目擊列最新 last_seen 的較大者——
        持續可見的靜態物體只有目擊列會被心跳刷新（身分列停在確認當下），
        只看身分列會把還在畫面裡的東西誤清。
        """
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT oi.object_id
                FROM object_identities oi
                LEFT JOIN (
                    SELECT object_id, MAX(last_seen) AS latest
                    FROM sightings WHERE object_id IS NOT NULL GROUP BY object_id
                ) s ON s.object_id = oi.object_id
                WHERE COALESCE(oi.last_seen, '') < ? AND COALESCE(s.latest, '') < ?
                """,
                (cutoff, cutoff),
            ).fetchall()
        return [r[0] for r in rows]

    def set_sighting_object(self, sighting_id: str, object_id: str) -> bool:
        """回傳是否真的更新到列——目擊列由 StorageWriter 非同步批次寫入，
        嵌入執行緒可能先到一步，呼叫端需據此重試。"""
        with self._conn() as conn:
            cur = conn.execute("UPDATE sightings SET object_id = ? WHERE sighting_id = ?",
                               (object_id, sighting_id))
            return cur.rowcount > 0

    def memory_rows(self, days: int = 7, object_id: str | None = None,
                    label_like: str | None = None) -> list[dict]:
        """情節建構的原料：依時間排序的（身分, 地點, 起訖）列。"""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        sql = ("SELECT object_id, label, location, first_seen, last_seen FROM sightings "
               "WHERE object_id IS NOT NULL AND last_seen >= ?")
        params: list = [cutoff]
        if object_id:
            sql += " AND object_id = ?"
            params.append(object_id)
        if label_like:
            sql += " AND label LIKE ?"
            params.append(f"%{label_like}%")
        sql += " ORDER BY first_seen"
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {"object_id": r[0], "label": r[1], "location": r[2],
             "first_seen": r[3], "last_seen": r[4]}
            for r in rows
        ]

    def latest_thumb_for_object(self, object_id: str) -> bytes | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT thumb FROM sightings WHERE object_id = ? AND thumb IS NOT NULL "
                "ORDER BY last_seen DESC LIMIT 1",
                (object_id,),
            ).fetchone()
        return row[0] if row else None

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

    _PERSONAL_COLS = ("item_id, name, label, n_photos, created_at, thumb, "
                      "last_seen, last_location, last_score, last_thumb")

    @staticmethod
    def _personal_row_to_dict(r) -> dict:
        return {
            "item_id": r[0], "name": r[1], "label": r[2], "n_photos": r[3],
            "created_at": r[4], "thumb": r[5], "last_seen": r[6],
            "last_location": r[7], "last_score": r[8], "last_thumb": r[9],
        }

    def personal_items(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT {self._PERSONAL_COLS} FROM personal_items "
                "ORDER BY created_at DESC"
            ).fetchall()
        return [self._personal_row_to_dict(r) for r in rows]

    def personal_labels(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT DISTINCT label FROM personal_items").fetchall()
        return [r[0] for r in rows]

    def get_personal_item(self, item_id: str) -> dict | None:
        with self._conn() as conn:
            r = conn.execute(
                f"SELECT {self._PERSONAL_COLS} FROM personal_items WHERE item_id = ?",
                (item_id,),
            ).fetchone()
        return self._personal_row_to_dict(r) if r else None

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
