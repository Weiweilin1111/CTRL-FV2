"""向量索引重建工具 —— 由 SQLite 縮圖重算 CLIP 嵌入，重建兩個向量索引。

SQLite 是事實來源，向量索引（data/vectors、data/personal_vectors）可隨時重建：
- 歷史目擊索引：sightings 表的縮圖 → 語意搜尋（/semantic）
- 個人物品索引：personal_photos 表的參考照縮圖 → 「我的物品」比對

使用時機：向量索引檔損毀、換了 CLIP 模型、或索引與資料庫明顯不同步。

用法（務必先停止後端，否則執行中的行程會覆寫剛重建的索引）：
    python -X utf8 tools\\rebuild_vectors.py

注意：縮圖解析度較低（預設 160px），重建後的個人物品向量精度略遜於
原始註冊照片；認不出時補拍幾張參考照即可。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ctrlf.config import ROOT_DIR, load_config  # noqa: E402
from ctrlf.storage.sqlite_repo import Repository  # noqa: E402
from ctrlf.storage.vector_store import NumpyVectorStore  # noqa: E402
from ctrlf.vision.embedder import create_embedder  # noqa: E402

BATCH = 16


def _decode(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    return img if img is not None and img.size > 0 else None


def _fresh_store(base_path: Path) -> NumpyVectorStore:
    """刪掉既有索引檔後建立全新索引（重建 = 從零開始，不保留孤兒向量）。"""
    for p in (base_path.with_suffix(".npz"), base_path.with_suffix(".json")):
        p.unlink(missing_ok=True)
    return NumpyVectorStore(base_path)


def _encode_batches(embedder, entries: list[tuple[str, dict, np.ndarray]],
                    store: NumpyVectorStore) -> int:
    done = 0
    for i in range(0, len(entries), BATCH):
        chunk = entries[i:i + BATCH]
        vectors = embedder.encode_images([img for _, _, img in chunk])
        if vectors is None or len(vectors) != len(chunk):
            print(f"  警告：第 {i // BATCH + 1} 批編碼失敗，略過 {len(chunk)} 筆")
            continue
        for (vid, payload, _), vec in zip(chunk, vectors):
            store.upsert(vid, vec, payload)
        done += len(chunk)
        print(f"  {done}/{len(entries)} ...", end="\r")
    store.flush()
    print()
    return done


def main() -> None:
    cfg = load_config()
    data_dir = ROOT_DIR / cfg.storage.data_dir
    if not (data_dir / "ctrlf.db").exists():
        sys.exit(f"找不到資料庫 {data_dir / 'ctrlf.db'}——沒有可重建的資料。")

    repo = Repository(data_dir / "ctrlf.db")
    embedder = create_embedder(cfg.embedder)
    if not embedder.enabled:
        sys.exit("CLIP 嵌入器不可用（embedder.enabled=false 或載入失敗），無法重建。")

    # ---- 歷史目擊索引 ----
    rows = repo.sightings_with_thumbs()
    entries = []
    for r in rows:
        img = _decode(r["thumb"])
        if img is None:
            continue
        entries.append((
            r["sighting_id"],
            {"label": r["label"], "display": f'{r["label"]}-{r["instance"]}',
             "location": r["location"] or "unknown", "last_seen": r["last_seen"]},
            img,
        ))
    print(f"歷史目擊：{len(rows)} 列，{len(entries)} 張可用縮圖")
    store = _fresh_store(data_dir / "vectors")
    n1 = _encode_batches(embedder, entries, store)

    # ---- 個人物品索引 ----
    entries = []
    for item in repo.personal_items():
        payload = {"item_id": item["item_id"], "name": item["name"], "label": item["label"]}
        for ph in repo.personal_photos(item["item_id"]):
            img = _decode(ph["thumb"])
            if img is None:
                continue
            entries.append((f'{item["item_id"]}#{ph["photo_id"]}', dict(payload), img))
    print(f"個人物品參考照：{len(entries)} 張")
    personal_store = _fresh_store(data_dir / "personal_vectors")
    n2 = _encode_batches(embedder, entries, personal_store)

    print(f"重建完成：歷史目擊 {n1} 筆、個人物品 {n2} 筆。可重新啟動後端。")


if __name__ == "__main__":
    main()
