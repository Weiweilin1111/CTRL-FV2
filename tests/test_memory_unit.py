"""視覺記憶層離線單元測試：身分合併/新建、情節建構、習慣統計（不需後端）。"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctrlf.config import MemoryConfig  # noqa: E402
from ctrlf.runtime.identity import IdentityManager  # noqa: E402
from ctrlf.runtime.memory import MemoryService  # noqa: E402
from ctrlf.storage.sqlite_repo import Repository  # noqa: E402

tmp = Path(tempfile.mkdtemp())
repo = Repository(tmp / "mem.db")
cfg = MemoryConfig()
identity = IdentityManager(repo, cfg)


def vec(seed: int, jitter: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=512).astype(np.float32)
    if jitter:
        v = v + jitter * np.random.default_rng(seed + 999).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


# 1) 相似觀測（同類別）合併為同一身分
base = vec(1)
oid_a, disp_a = identity.assign(base, "cup", "desk", "2026-06-13 09:00:00")
oid_b, disp_b = identity.assign(vec(1, jitter=0.15), "cup", "desk", "2026-06-13 09:30:00")
assert oid_a == oid_b and disp_a == "cup#1", (oid_a, oid_b, disp_a)
print("1) 相似觀測合併 OK:", disp_a)

# 2) 不相似觀測（同類別）建立新身分
oid_c, disp_c = identity.assign(vec(42), "cup", "sofa", "2026-06-13 10:00:00")
assert oid_c != oid_a and disp_c == "cup#2", (oid_c, disp_c)
print("2) 不相似觀測新建 OK:", disp_c)

# 3) 類別閘控：不同 label 即使向量相同也不得合併
oid_d, disp_d = identity.assign(base, "wallet", "desk", "2026-06-13 10:05:00")
assert oid_d != oid_a and disp_d == "wallet#1"
print("3) 類別閘控 OK:", disp_d)

# 4) 重啟持久性：重建引擎後仍合併到原身分
identity2 = IdentityManager(repo, cfg)
oid_e, disp_e = identity2.assign(vec(1, jitter=0.1), "cup", "desk", "2026-06-13 14:00:00")
assert oid_e == oid_a, "重啟後身分應持續"
print("4) 跨重啟身分持續 OK")

# 5) 情節建構：同地點近時合併、跨地點/長間隔切段
memory = MemoryService(repo, cfg, retention_days=7)
from datetime import datetime, timedelta  # noqa: E402

now = datetime.now()


def fmt(offset_min):
    return (now + timedelta(minutes=offset_min)).strftime("%Y-%m-%d %H:%M:%S")


rows = [
    ("s1", "desk", fmt(-300), fmt(-290)),
    ("s2", "desk", fmt(-285), fmt(-270)),   # gap 5min -> 同情節
    ("s3", "sofa", fmt(-200), fmt(-180)),   # 換地點 -> 新情節
    ("s4", "desk", fmt(-100), fmt(-90)),    # gap 大 -> 新情節
]
for sid, loc, t0, t1 in rows:
    from ctrlf.core.models import SightingEvent
    repo.upsert_sightings([SightingEvent(
        sighting_id=sid, label="cup", instance=1, location=loc,
        bbox=(0, 0, 10, 10), confidence=0.9, first_seen=t0, last_seen=t1)])
    repo.set_sighting_object(sid, oid_a)

eps = memory.episodes(object_id=oid_a)
assert len(eps) == 3, [e.to_dict() for e in eps]
print("5) 情節建構 OK: 4 筆目擊 ->", len(eps), "段情節")

# 6) 習慣統計：依停留時間加權
habits = memory.habits(oid_a)
assert habits and habits[0]["location"] == "desk", habits
total_pct = sum(h["pct"] for h in habits)
assert 0.99 <= total_pct <= 1.01
print("6) 習慣統計 OK:", [(h["location"], h["pct"]) for h in habits])

# 7) 回溯查詢
recall = memory.recall("cup")
assert recall and recall[0]["latest_episode"] is not None
print("7) 回溯查詢 OK:", recall[0]["display"], "->",
      recall[0]["latest_episode"]["location"])

# 8) 身分 TTL 清理：無近期目擊的過期身分移除；
#    有近期目擊列的身分（cup#1，模擬持續可見的靜態物體）與新身分保留
oid_f, _ = identity2.assign(vec(77), "keys", "desk",
                            now.strftime("%Y-%m-%d %H:%M:%S"))
before = identity2.count()
removed = identity2.prune(7)
kept = {i["object_id"] for i in repo.identities_meta()}
assert oid_a in kept, "有近期目擊的身分不得被清除"
assert oid_f in kept, "新身分不得被清除"
assert oid_c not in kept and oid_d not in kept, "過期且無目擊的身分應被清除"
assert identity2.count() == before - removed
print(f"8) 身分 TTL 清理 OK: 移除 {removed} 個過期身分，保留 {len(kept)} 個")
print("ALL MEMORY UNIT TESTS PASSED")
