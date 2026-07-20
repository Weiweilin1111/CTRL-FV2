"""Lexicon — 中文/別名 → 偵測詞彙的離線翻譯層。

YOLO-World 與 CLIP 的文字端都吃英文；本模組讓使用者直接輸入
「鑰匙」「我的錢包在哪」也能解析成 canonical 英文標籤。
解析順序：自訂同義詞（SQLite）→ 內建辭典 → 子字串比對（長詞優先）
→ 純英文直接通過；中文查無對應時回傳 None 交由 UI 引導補辭典。
"""
from __future__ import annotations

import threading

# 內建辭典：繁體為主、常見簡體與英文俗名一併收錄
_BUILTIN: dict[str, str] = {
    # 3C 與電子
    "手機": "cell phone", "手机": "cell phone", "電話": "cell phone",
    "iphone": "cell phone", "mobile": "cell phone", "phone": "cell phone",
    "smartphone": "cell phone",
    "筆電": "laptop", "筆記型電腦": "laptop", "笔记本电脑": "laptop", "電腦": "laptop",
    "平板": "tablet",
    "滑鼠": "mouse", "鼠标": "mouse",
    "鍵盤": "keyboard", "键盘": "keyboard",
    "耳機": "headphones", "耳机": "headphones", "藍牙耳機": "earphones", "耳塞": "earphones",
    "充電器": "charger", "充电器": "charger", "充電線": "charger", "傳輸線": "charger",
    "行動電源": "power bank", "充電寶": "power bank", "充电宝": "power bank",
    "遙控器": "remote control", "遥控器": "remote control",
    "手錶": "watch", "手表": "watch",
    "相機": "camera", "相机": "camera",
    # 隨身物品
    "鑰匙": "keys", "钥匙": "keys",
    "錢包": "wallet", "钱包": "wallet", "皮夾": "wallet", "皮夹": "wallet",
    "眼鏡": "glasses", "眼镜": "glasses",
    "墨鏡": "sunglasses", "太陽眼鏡": "sunglasses",
    "背包": "backpack", "書包": "backpack",
    "包包": "handbag", "手提包": "handbag", "提袋": "handbag",
    "雨傘": "umbrella", "雨伞": "umbrella", "傘": "umbrella",
    "口罩": "face mask",
    "卡片": "card", "悠遊卡": "card", "證件": "card",
    "帽子": "hat", "圍巾": "scarf", "手套": "gloves",
    # 文具與居家
    "筆": "pen", "原子筆": "pen", "鉛筆": "pencil",
    "剪刀": "scissors",
    "書": "book", "書本": "book",
    "筆記本": "notebook", "本子": "notebook",
    "水壺": "bottle", "水瓶": "bottle", "瓶子": "bottle", "寶特瓶": "bottle",
    "杯子": "cup", "馬克杯": "mug", "马克杯": "mug",
    "面紙": "tissue box", "面紙盒": "tissue box", "衛生紙": "tissue box",
    "梳子": "comb",
    "打火機": "lighter", "打火机": "lighter",
    "藥": "medicine bottle", "藥罐": "medicine bottle", "藥瓶": "medicine bottle",
    "藥盒": "pill box",
    "玩具": "toy", "娃娃": "stuffed toy", "玩偶": "stuffed toy",
}


class Lexicon:
    def __init__(self, custom: dict[str, str] | None = None):
        self._lock = threading.Lock()
        self._custom: dict[str, str] = {}
        self.reload(custom or {})

    def reload(self, custom: dict[str, str]) -> None:
        with self._lock:
            self._custom = {k.strip().lower(): v.strip().lower()
                            for k, v in custom.items() if k.strip() and v.strip()}

    def resolve(self, raw: str) -> tuple[str | None, str | None]:
        """回傳 (canonical 英文標籤, 來源)；來源 ∈ {custom, builtin, None}。

        中文且查無對應時回傳 (None, None)。
        """
        q = (raw or "").strip().lower()
        if not q:
            return "", None
        with self._lock:
            custom = dict(self._custom)

        if q in custom:
            return custom[q], "custom"
        if q in _BUILTIN:
            return _BUILTIN[q], "builtin"

        # 子字串比對（處理「我的鑰匙」「鑰匙在哪」），長詞優先避免誤截
        for table, via in ((custom, "custom"), (_BUILTIN, "builtin")):
            for key in sorted(table, key=len, reverse=True):
                if key in q:
                    return table[key], via

        if q.isascii():
            return q, None  # 英文視為 canonical，直接通過
        return None, None

    @property
    def builtin_size(self) -> int:
        return len(_BUILTIN)
