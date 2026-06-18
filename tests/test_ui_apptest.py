"""Streamlit AppTest（請於專案根目錄執行，且後端需運行中）：真實執行 ui/app.py 並模擬互動。"""
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("ui/app.py", default_timeout=90)
at.run()
assert not at.exception, f"初次渲染例外: {at.exception}"
print(f"1) 初次渲染 OK（markdown={len(at.markdown)}, button={len(at.button)}, "
      f"slider={len(at.slider)}, expander 等元素正常）")

# 中文搜尋互動
at.text_input(key="query_input").input("鑰匙").run()
assert not at.exception, f"搜尋互動例外: {at.exception}"
assert at.session_state["query"] == "keys", at.session_state["query"]
print("2) 中文搜尋互動 OK: 鑰匙 -> keys")

# 滑桿調參
at.slider(key="sl_conf").set_value(0.30).run()
assert not at.exception
print("3) 滑桿調參 OK")

# 未解析中文應顯示警告而非崩潰
at.text_input(key="query_input").input("外星雷射砲").run()
assert not at.exception
warnings = [w.value for w in at.warning]
assert any("外星雷射砲" in w for w in warnings), warnings
print("4) 未解析中文警告 OK")

# 清空查詢
at.text_input(key="query_input").input("").run()
assert not at.exception
print("5) 清空查詢 OK")

# 我的物品「找它」：fragment 內按鈕改主畫面查詢（曾因直接改 widget 狀態而炸）
import base64

import cv2
import numpy as np
import requests

img = np.full((300, 400, 3), 200, np.uint8)
cv2.circle(img, (200, 150), 80, (50, 50, 200), -1)
_, buf = cv2.imencode(".jpg", img)
b64 = base64.b64encode(buf.tobytes()).decode("ascii")
item = requests.post("http://127.0.0.1:8000/items",
                     json={"name": "測試找它", "label": "toy", "images": [b64]},
                     timeout=60).json()
try:
    at2 = AppTest.from_file("ui/app.py", default_timeout=90)
    at2.run()
    at2.button(key=f"find_{item['item_id']}").click().run()
    assert not at2.exception, f"找它例外: {at2.exception}"
    assert at2.session_state["query"] == "toy", at2.session_state["query"]
    print("6) 我的物品「找它」OK: query -> toy")
finally:
    requests.delete(f"http://127.0.0.1:8000/items/{item['item_id']}", timeout=10)
print("ALL UI TESTS PASSED")

