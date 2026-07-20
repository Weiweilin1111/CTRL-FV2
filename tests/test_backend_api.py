"""後端全端點回歸測試（後端需運行中）。"""
import base64
import json

import cv2
import numpy as np
import requests

API = "http://127.0.0.1:8000"
PASS = []


def check(name, cond, info=""):
    PASS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info else ""))


def img_b64(color) -> str:
    img = np.full((480, 640, 3), 235, np.uint8)
    cv2.rectangle(img, (180, 120), (460, 360), color, -1)
    ok, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode("ascii")


S = requests.Session()

# ---- system ----
st_ = S.get(API + "/system/stats", timeout=10).json()
check("stats.version=3.2.1", st_.get("version") == "3.2.1")
check("stats.detector", st_.get("detector", {}).get("model", "").endswith(".pt"))
check("stats.gpu", "gpu" in st_)
check("stats.storage", "storage_mb" in st_)

cfg0 = S.get(API + "/system/settings", timeout=5).json()
rr = S.post(API + "/system/settings", json={"conf": 5.0}, timeout=5)
check("settings reject out-of-range (422)", rr.status_code == 422)
patched = S.post(API + "/system/settings", json={"conf": 0.5}, timeout=5).json()
check("settings apply valid", patched["conf"] == 0.5)
S.post(API + "/system/settings", json={"conf": cfg0["conf"]}, timeout=5)

# ---- search / lexicon ----
r = S.post(API + "/search", json={"query": "cup"}, timeout=10).json()
check("search english", r["resolved"] and r["query"] == "cup")
r = S.post(API + "/search", json={"query": "鑰匙"}, timeout=10).json()
check("search zh builtin", r["resolved"] and r["query"] == "keys" and r["via"] == "builtin")
r = S.post(API + "/search", json={"query": "我的手機放哪了"}, timeout=10).json()
check("search zh sentence", r["resolved"] and r["query"] == "cell phone")
r = S.post(API + "/search", json={"query": "外星雷射砲"}, timeout=10).json()
check("search zh unresolved", r["resolved"] is False)
r = S.post(API + "/search", json={"query": ""}, timeout=10).json()
check("search empty clears", r["resolved"] and r["query"] == "")

# ---- state / history / semantic ----
state = S.get(API + "/state", timeout=5).json()
check("state shape", "tracks" in state and "stats" in state and "query" in state)
hist = S.get(API + "/history", params={"query": "bottle", "limit": 3}, timeout=5).json()
check("history records", isinstance(hist.get("records"), list))
sem = S.get(API + "/semantic", params={"query": "馬克杯", "k": 3}, timeout=15).json()
check("semantic zh visual", sem.get("mode") in ("visual", "label"), f"mode={sem.get('mode')}")
sem2 = S.get(API + "/semantic", params={"query": "外星雷射砲"}, timeout=10).json()
check("semantic unresolved", sem2.get("mode") == "unresolved")

# ---- tags ----
S.post(API + "/tags/fixed", json={"tag": "test-fixed-tag"}, timeout=10)
tags = S.get(API + "/tags", timeout=5).json()
check("fixed tag added", "test-fixed-tag" in tags["fixed"])
S.delete(API + "/tags/fixed/test-fixed-tag", timeout=10)
tags = S.get(API + "/tags", timeout=5).json()
check("fixed tag removed", "test-fixed-tag" not in tags["fixed"])

# ---- aliases ----
S.post(API + "/aliases", json={"alias": "測試詞", "label": "cup"}, timeout=5)
al = S.get(API + "/aliases", timeout=5).json()
check("alias added", al["custom"].get("測試詞") == "cup")
r = S.post(API + "/search", json={"query": "測試詞"}, timeout=10).json()
check("alias resolves", r["query"] == "cup" and r["via"] == "custom")
S.delete(API + "/aliases/測試詞", timeout=5)
check("alias removed", "測試詞" not in S.get(API + "/aliases", timeout=5).json()["custom"])

# ---- items full cycle ----
g1, g2, red = img_b64((60, 200, 60)), img_b64((70, 205, 65)), img_b64((40, 40, 220))
it = S.post(API + "/items", json={"name": "回歸測試物", "label": "toy",
                                  "images": [g1, g2]}, timeout=60).json()
iid = it["item_id"]
check("item register", it["n_photos"] == 2)
detail = S.get(f"{API}/items/{iid}", timeout=5).json()
check("item detail photos", len(detail["photos"]) == 2 and all(p["thumb"] for p in detail["photos"]))
add = S.post(f"{API}/items/{iid}/photos", json={"images": [g1]}, timeout=60).json()
check("item add photo", add["added"] == 1)
t = S.post(f"{API}/items/{iid}/test", json={"image": g2}, timeout=60).json()
check("item test score", 0.0 < t["score"] <= 1.0, f"score={t['score']}")
pid = S.get(f"{API}/items/{iid}", timeout=5).json()["photos"][0]["photo_id"]
rr = S.delete(f"{API}/items/{iid}/photos/{pid}", timeout=10)
check("item remove photo", rr.status_code == 200)
rr = S.patch(f"{API}/items/{iid}", json={"name": "改名"}, timeout=5)
check("item rename", rr.status_code == 200)
bad = S.post(API + "/items", json={"name": "x", "label": "中文類別", "images": [g1]}, timeout=10)
check("item zh label rejected", bad.status_code == 400)
rr = S.delete(f"{API}/items/{iid}", timeout=10)
check("item delete", rr.status_code == 200)

# ---- memory（視覺記憶層） ----
mem = S.get(API + "/memory/objects", timeout=10).json()
check("memory objects shape", isinstance(mem.get("objects"), list) and "identities" in mem)
rec = S.get(API + "/memory/recall", params={"query": "杯子"}, timeout=10).json()
check("memory recall zh", rec.get("label") == "cup" and isinstance(rec.get("objects"), list))
if mem["objects"]:
    oid = mem["objects"][0]["object_id"]
    tl = S.get(API + "/memory/timeline", params={"object_id": oid}, timeout=10).json()
    check("memory timeline", isinstance(tl.get("episodes"), list) and isinstance(tl.get("habits"), list))
else:
    check("memory timeline", S.get(API + "/memory/timeline",
                                   params={"object_id": "obj_none"}, timeout=10).status_code == 404)

# ---- stream / ws ----
r = S.get(API + "/stream/mjpeg", stream=True, timeout=5)
chunk = next(r.iter_content(8192))
check("mjpeg stream", chunk.startswith(b"--frame"))
r.close()
try:
    import websocket
    ws = websocket.create_connection("ws://127.0.0.1:8000/ws/state", timeout=5)
    msg = json.loads(ws.recv())
    ws.close()
    check("websocket state", "tracks" in msg and "stats" in msg)
except Exception as e:  # noqa: BLE001
    check("websocket state", False, str(e))

failed = [n for n, ok in PASS if not ok]
print(f"\n{len(PASS) - len(failed)}/{len(PASS)} PASSED" + (f" | FAILED: {failed}" if failed else ""))

