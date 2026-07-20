"""CtrlF — Streamlit 前端（v3.2）。

設計原則：
- UI 是純消費者：只透過 REST/WS 與後端溝通，零業務邏輯。
- 影像走 MJPEG（iframe 內自動重連 + 骨架屏動畫），效能資訊走 WebSocket HUD，
  兩者都不觸發 Streamlit rerun —— 畫面永不因 UI 互動而中斷。
- 局部更新一律用 st.fragment(run_every=...)；訊息層級與文案以非工程師可讀為準。
"""
from __future__ import annotations

import base64
import html
import io

import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

API = "http://127.0.0.1:8000"
TIMEOUT = 2.5
UI_VERSION = "3.2.1"  # 必須與後端 ctrlf.__version__ 一致，否則提示重啟

# ---------------------------------------------------------------- API helpers


def api_get(path: str, **params) -> dict | None:
    try:
        r = requests.get(API + path, params=params or None, timeout=TIMEOUT)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None


def api_post(path: str, payload: dict | None = None) -> dict | None:
    try:
        r = requests.post(API + path, json=payload or {}, timeout=TIMEOUT)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None


def api_delete(path: str) -> None:
    try:
        requests.delete(API + path, timeout=TIMEOUT)
    except requests.RequestException:
        pass


def api_patch(path: str, payload: dict) -> dict | None:
    try:
        r = requests.patch(API + path, json=payload, timeout=TIMEOUT)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None


def api_post_long(path: str, payload: dict, timeout: float = 120.0):
    """長時間請求（照片上傳/CLIP 編碼）。回傳 (data, 錯誤訊息)。"""
    try:
        r = requests.post(API + path, json=payload, timeout=timeout)
        try:
            data = r.json()
        except ValueError:
            data = {}
        if r.status_code == 200:
            return data, None
        detail = data.get("detail") if isinstance(data, dict) else None
        return None, str(detail or f"伺服器錯誤 (HTTP {r.status_code})")
    except requests.RequestException as e:
        return None, f"無法連線後端：{e}"


def photo_to_b64(file) -> str:
    """手機原圖可達 10MB+：先校正 EXIF 方向並縮至 1280px 再上傳。"""
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    img.thumbnail((1280, 1280))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


AREA_ZH = {
    "bed": "床", "desk": "書桌", "table": "桌子", "nightstand": "床頭櫃",
    "shelf": "層架", "chair": "椅子", "cabinet": "櫃子", "floor": "地板",
    "sofa": "沙發", "windowsill": "窗台", "unknown": "未知區域",
}


def loc_zh(loc: str | None) -> str:
    return AREA_ZH.get(loc or "unknown", loc or "未知區域")


def esc(x) -> str:
    """使用者可控文字（物品名、標籤、查詢詞）進 unsafe_allow_html 前一律轉義。"""
    return html.escape(str(x if x is not None else ""))


@st.cache_data(ttl=5.0, show_spinner=False)
def fetch_memory_objects() -> dict:
    """記憶分頁的資料。Streamlit 每次互動都會重跑整個腳本——
    加 5 秒快取讓拉滑桿/搜尋不會反覆打到記憶端點。"""
    return api_get("/memory/objects") or {}


@st.cache_data(ttl=5.0, show_spinner=False)
def fetch_memory_timeline(object_id: str) -> dict:
    return api_get("/memory/timeline", object_id=object_id) or {}


# ---------------------------------------------------------------- page setup

st.set_page_config(
    page_title="CtrlF · 智慧尋物",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* ---- 版面骨架 ---- */
.block-container { padding-top: 2.5rem; padding-bottom: 2.5rem; max-width: 1380px; }
#MainMenu, [data-testid="stMainMenu"], footer { visibility: hidden; }
[data-testid="stAppDeployButton"], .stAppDeployButton { display: none; }
hr { border-color: #222b40 !important; }

/* ---- 品牌與標頭 ---- */
.hero-title { font-size: 32px; font-weight: 800; letter-spacing: .5px; line-height: 1.1;
  background: linear-gradient(90deg, #4ade80, #38bdf8); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; margin: 0; }
.hero-sub { color: #8b97b5; font-size: 13.5px; margin-top: 3px; }

/* ---- 區塊標題 ---- */
.sec-title { display: flex; align-items: center; gap: 8px; color: #dbe2f0;
  font-size: 16.5px; font-weight: 700; margin: 6px 0 10px; }
.sec-title .hint { color: #6b7794; font-size: 12px; font-weight: 400; margin-left: 4px; }

/* ---- 徽章 ---- */
.badge { display: inline-block; padding: 3px 12px; border-radius: 999px;
  font-size: 12px; font-weight: 600; margin-bottom: 8px; }
.badge.green { background: rgba(34,197,94,.14); color: #4ade80; border: 1px solid rgba(34,197,94,.35); }
.badge.blue  { background: rgba(59,130,246,.14); color: #60a5fa; border: 1px solid rgba(59,130,246,.35); }
.badge.gray  { background: rgba(148,163,184,.12); color: #94a3b8; border: 1px solid rgba(148,163,184,.3); }
.badge.red   { background: rgba(244,63,94,.14); color: #fb7185; border: 1px solid rgba(244,63,94,.35); }

/* ---- 卡片 ---- */
.ctrl-card { background: #151b29; border: 1px solid #28324a; border-radius: 14px;
  padding: 13px 15px; margin-bottom: 10px; transition: border-color .15s; }
.ctrl-card:hover { border-color: #3a4a6e; }
.ctrl-card.hit { border-left: 3px solid #22c55e; }
.ctrl-card .row { display: flex; gap: 12px; align-items: center; }
.ctrl-card .t { color: #e8ecf6; font-weight: 650; font-size: 15px; }
.ctrl-card .s { color: #8b97b5; font-size: 12.5px; margin-top: 3px; line-height: 1.6; }
.thumb { width: 80px; height: 60px; object-fit: cover; border-radius: 9px;
  border: 1px solid #2c3650; flex-shrink: 0; }
.thumb-empty { display: flex; align-items: center; justify-content: center;
  background: #0f1420; color: #44506e; font-size: 20px; }

/* ---- 新手引導步驟 ---- */
.step { display: flex; gap: 10px; align-items: flex-start; color: #9aa6c0;
  margin-top: 10px; font-size: 13.5px; line-height: 1.55; }
.step .num { background: rgba(34,197,94,.13); color: #4ade80; border: 1px solid rgba(34,197,94,.3);
  min-width: 22px; height: 22px; border-radius: 999px; display: flex; align-items: center;
  justify-content: center; font-weight: 700; font-size: 12px; flex-shrink: 0; margin-top: 1px; }

/* ---- 元件微調 ---- */
div[data-testid="stMetricValue"] { font-size: 1.3rem; }
div[data-testid="stMetricLabel"] { color: #8b97b5; }
.stButton > button { border-radius: 10px; }
button[data-baseweb="tab"] { font-size: 15px; padding: 8px 18px; }
div[data-testid="stExpander"] details { border-radius: 12px; border-color: #28324a; }
.stProgress > div > div { border-radius: 999px; }
.foot { text-align: center; color: #57617c; font-size: 12px; margin-top: 28px; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------- backend handshake & state

remote = api_get("/system/settings")
if remote is None:
    st.markdown('<p class="hero-title">CtrlF</p>', unsafe_allow_html=True)
    st.error("無法連接後端服務（127.0.0.1:8000）。請執行 `start.ps1`（或 `python run_api.py`）後再重試。")
    if st.button("重試連線"):
        st.rerun()
    st.stop()

if "seeded" not in st.session_state:
    st.session_state.update(
        seeded=True,
        query=remote.get("query", ""),
        query_input=remote.get("query", ""),
        sl_conf=float(remote.get("conf", 0.2)),
        sl_sens=float(remote.get("motion_sensitivity", 0.5)),
        sl_fps=float(remote.get("max_infer_fps", 5.0)),
        sl_show=bool(remote.get("show_all", False)),
    )


def push_settings() -> None:
    api_post("/system/settings", {
        "conf": st.session_state.sl_conf,
        "motion_sensitivity": st.session_state.sl_sens,
        "max_infer_fps": st.session_state.sl_fps,
        "show_all": st.session_state.sl_show,
    })


def set_query(q: str) -> None:
    q = (q or "").strip()
    res = api_post("/search", {"query": q})
    st.session_state["resolve_info"] = res
    if res is None:  # 後端離線時退化為本地狀態
        st.session_state["query"] = q.lower()
    elif res.get("resolved"):
        st.session_state["query"] = res.get("query", "")


def apply_query_input() -> None:
    st.session_state["query_input"] = st.session_state.query_input.strip().lower()
    set_query(st.session_state.query_input)


def apply_chip() -> None:
    sel = st.session_state.get("chip")
    if sel:
        tag = sel.split(" ", 1)[1]
        st.session_state["query_input"] = tag
        set_query(tag)


def clear_query() -> None:
    st.session_state["query_input"] = ""
    set_query("")


# Fragment（我的物品/建議按鈕）不能直接改主畫面 widget 的狀態，
# 改走暫存鍵：fragment 寫入 pending_query 後觸發全頁重繪，
# 這裡在搜尋框實體化「之前」套用，繞過 StreamlitAPIException。
_pending = st.session_state.pop("pending_query", None)
if _pending is not None:
    st.session_state["query_input"] = _pending
    set_query(_pending)


tags = api_get("/tags") or {"fixed": [], "recent": []}
fixed_tags: list[str] = tags.get("fixed", [])
recent_tags: list[str] = [t for t in tags.get("recent", []) if t not in fixed_tags]

# ------------------------------------------------------------ render helpers


def _thumb_html(b64: str | None) -> str:
    if b64:
        return f'<img class="thumb" src="data:image/jpeg;base64,{b64}">'
    return '<div class="thumb thumb-empty">—</div>'


def _card(html: str, tone: str = "") -> None:
    st.markdown(f'<div class="ctrl-card {tone}">{html}</div>', unsafe_allow_html=True)


def _cards_grid(cards: list[str], ncols: int = 2, tone: str = "") -> None:
    """卡片以多欄網格排列（全寬版面下避免單欄拉太長）。"""
    cols = st.columns(ncols)
    for i, html in enumerate(cards):
        with cols[i % ncols]:
            _card(html, tone)


def _section(icon: str, title: str, hint: str = "") -> None:
    hint_html = f'<span class="hint">{hint}</span>' if hint else ""
    icon_html = f"{icon} " if icon else ""
    st.markdown(f'<div class="sec-title">{icon_html}{title}{hint_html}</div>',
                unsafe_allow_html=True)


# -------------------------------------------------------------------- sidebar

with st.sidebar:
    _section("📈", "系統監控")

    @st.fragment(run_every=2.5)
    def stats_panel() -> None:
        data = api_get("/system/stats")
        if not data:
            st.markdown('<span class="badge red">● 後端離線</span>', unsafe_allow_html=True)
            return
        api_ver = data.get("version")
        if api_ver != UI_VERSION:
            st.warning(f"後端版本（{api_ver or '3.0 以前'}）與介面（{UI_VERSION}）不一致，"
                       "後端還在跑舊程式碼 —— 請關閉後重新執行 start.ps1。")
        c1, c2 = st.columns(2)
        c1.metric("串流畫面", f"{data.get('capture_fps', 0)} fps")
        c2.metric("AI 推論", f"{data.get('infer_fps', 0)} fps")
        c1.metric("推論延遲", f"{data.get('infer_ms', 0)} ms")
        c2.metric("追蹤中", f"{data.get('tracked', 0)} 件")
        gpu = data.get("gpu")
        if gpu:
            used, total = gpu["vram_used_mb"], max(1, gpu["vram_total_mb"])
            st.progress(min(1.0, used / total), text=f"顯示卡記憶體 {used} / {total} MB")
        det = data.get("detector", {})
        st.caption(
            f"{det.get('model', '?')} · {det.get('device', '?').upper()} · {det.get('imgsz', '?')}px"
            f"  \n視覺特徵 {data.get('vectors', 0)} 筆 · 畫面動態 {data.get('motion_ratio', 0):.1%}"
            f"  \n已用 {data.get('storage_mb', 0)} MB（紀錄保留 {data.get('retention_days', 7)} 天）"
        )

    stats_panel()

    st.divider()
    _section("⚙️", "偵測調校", "拉桿即時生效")
    st.slider("偵測信心門檻", 0.05, 0.90, step=0.05, key="sl_conf",
              on_change=push_settings,
              help="AI 要多有把握才算「看到」。調低找得更積極但可能誤認；調高更嚴格。")
    st.slider("畫面變化敏感度", 0.0, 1.0, step=0.05, key="sl_sens",
              on_change=push_settings,
              help="畫面有變化才啟動 AI 分析。調高反應更快、調低更省電。")
    st.slider("AI 分析頻率上限", 1.0, 12.0, step=0.5, key="sl_fps",
              on_change=push_settings,
              help="每秒最多分析幾次。畫面流暢度不受此影響。")
    st.toggle("顯示所有偵測到的物件", key="sl_show", on_change=push_settings,
              help="預設只框出你正在找的目標；打開可看到 AI 眼中的全部物件。")

    st.divider()
    _section("📌", "常駐追蹤")
    st.caption("常駐物件會持續記錄出現位置，但畫面不特別框出。")
    def _add_fixed_tag() -> None:
        tag = st.session_state.fixed_input.strip()
        if tag:
            api_post("/tags/fixed", {"tag": tag})
            st.session_state.fixed_input = ""  # callback 內可清空輸入框

    st.text_input("新增常駐物件（英文）", key="fixed_input",
                  placeholder="例如 medicine bottle")
    st.button("➕ 加入常駐", width="stretch", on_click=_add_fixed_tag)
    for tag in fixed_tags:
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"<div style='padding-top:6px'>⭐ {esc(tag)}</div>", unsafe_allow_html=True)
        if c2.button("✕", key=f"del_{tag}"):
            api_delete(f"/tags/fixed/{requests.utils.quote(tag, safe='')}")
            st.rerun()

    st.divider()
    with st.expander("中文同義詞辭典"):
        al = api_get("/aliases") or {"custom": {}, "builtin_count": 0}
        st.caption(f"已內建 {al.get('builtin_count', 0)} 個常用詞（鑰匙、錢包、充電器…）。"
                   "家人有自己的講法？在這裡加上對應。")
        def _add_alias() -> None:
            k = st.session_state.al_key.strip()
            v = st.session_state.al_val.strip()
            if k and v:
                api_post("/aliases", {"alias": k, "label": v})
                st.session_state.al_key = ""
                st.session_state.al_val = ""
            else:
                st.session_state["al_msg"] = "兩欄都要填。"

        ac1, ac2 = st.columns(2)
        ac1.text_input("你的講法", key="al_key", placeholder="充電頭")
        ac2.text_input("英文標籤", key="al_val", placeholder="charger")
        st.button("➕ 新增同義詞", width="stretch", on_click=_add_alias)
        if st.session_state.pop("al_msg", None):
            st.warning("兩欄都要填。")
        for k, v in (al.get("custom") or {}).items():
            rc1, rc2 = st.columns([5, 1])
            rc1.markdown(f"<div style='padding-top:6px'>{esc(k)} → {esc(v)}</div>",
                         unsafe_allow_html=True)
            if rc2.button("✕", key=f"dal_{k}"):
                api_delete(f"/aliases/{requests.utils.quote(k, safe='')}")
                st.rerun()

# --------------------------------------------------------------------- header

head_l, head_r = st.columns([4, 1.4])
with head_l:
    st.markdown('<p class="hero-title">CtrlF</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">即時智慧尋物 — 看著畫面找、查最後位置、認得你的東西</p>',
                unsafe_allow_html=True)
with head_r:
    @st.fragment(run_every=5.0)
    def status_pill() -> None:
        data = api_get("/system/stats")
        if data:
            device = (data.get("detector") or {}).get("device", "?").upper()
            ver = data.get("version", "?")
            st.markdown(f'<div style="text-align:right;padding-top:16px">'
                        f'<span class="badge green">● 運作中 · {device} · v{ver}</span></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:right;padding-top:16px">'
                        '<span class="badge red">● 後端離線</span></div>',
                        unsafe_allow_html=True)

    status_pill()

# --------------------------------------------------------------------- search

_section("", "你要找什麼？", "中文、英文都可以")
in_col, clr_col = st.columns([8, 1])
with in_col:
    st.text_input(
        "搜尋", key="query_input", on_change=apply_query_input,
        placeholder="例如：鑰匙、錢包、cup ⋯ 輸入後按 Enter",
        label_visibility="collapsed",
    )
with clr_col:
    st.button("✕ 清除", width="stretch", on_click=clear_query,
              disabled=not st.session_state.get("query"))

_info = st.session_state.get("resolve_info")
if _info:
    if not _info.get("resolved", True):
        st.warning(
            f"還聽不懂「{_info['original']}」這個詞。請到側欄「中文同義詞辭典」教它"
            f"（例如：{_info['original']} → keys），或改用英文。"
        )
    elif _info.get("via") and _info.get("original", "").strip().lower() != _info.get("query", ""):
        src = "自訂同義詞" if _info["via"] == "custom" else "內建辭典"
        st.caption(f"已自動翻譯：「{_info['original']}」→ **{_info['query']}**（{src}）")

chip_options = [f"⭐ {t}" for t in fixed_tags] + [f"🕒 {t}" for t in recent_tags]
if chip_options:
    st.pills("快速搜尋", chip_options, selection_mode="single",
             key="chip", on_change=apply_chip, label_visibility="collapsed")
    st.caption("⭐ 常駐物件 · 🕒 最近搜尋過")

# ----------------------------------------------------------------------- tabs

tab_live, tab_memory, tab_manage = st.tabs(["即時尋物", "記憶時間軸", "物品註冊管理"])

with tab_live:
    _section("", "即時畫面", "找到的目標會以綠框鎖定")
    stream_html = """
<!DOCTYPE html><html><head><style>
  body { margin:0; background:transparent; font-family:'Segoe UI',sans-serif; }
  .wrap { position:relative; width:100%; max-width:860px; margin:0 auto; aspect-ratio:16/9;
          border-radius:16px; overflow:hidden;
          background:#0b0f17; border:1px solid #232b3d; box-shadow:0 8px 30px rgba(0,0,0,.45); }
  #feed { width:100%; height:100%; object-fit:contain; display:block; }
  .sk { position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
        color:#5b6b8c; font-size:15px; letter-spacing:1px;
        background:linear-gradient(110deg,#101624 8%,#1a2336 18%,#101624 33%);
        background-size:200% 100%; animation:sh 1.4s linear infinite; }
  @keyframes sh { to { background-position-x:-200%; } }
  #hud { position:absolute; right:10px; bottom:10px; display:none; padding:6px 12px;
         background:rgba(8,12,20,.72); backdrop-filter:blur(6px); color:#9fe8b2;
         font:12px Consolas,monospace; border-radius:999px;
         border:1px solid rgba(120,255,160,.25); }
</style></head><body>
<div class="wrap">
  <div class="sk" id="sk">正在連接攝影機串流…</div>
  <img id="feed" alt="">
  <div id="hud"></div>
</div>
<script>
  const API = "__API__";
  const img = document.getElementById('feed'), sk = document.getElementById('sk');
  function connect() { img.src = API + "/stream/mjpeg?t=" + Date.now(); }
  img.onload = () => { sk.style.display = 'none'; };
  img.onerror = () => { sk.style.display = 'flex'; setTimeout(connect, 2000); };
  connect();
  function ws() {
    try {
      const s = new WebSocket(API.replace('http', 'ws') + "/ws/state");
      s.onmessage = (e) => {
        const d = JSON.parse(e.data), st = d.stats || {};
        const hud = document.getElementById('hud');
        hud.style.display = 'block';
        hud.textContent = '串流 ' + (st.capture_fps || 0) + ' fps · 推論 ' + (st.infer_fps || 0)
          + ' fps · ' + (st.infer_ms || 0) + ' ms · 追蹤 ' + (d.tracks || []).length;
      };
      s.onclose = () => setTimeout(ws, 3000);
    } catch (e) { setTimeout(ws, 3000); }
  }
  ws();
</script></body></html>
""".replace("__API__", API)
    components.html(stream_html, height=500)

    _section("", "搜尋結果")

    @st.fragment(run_every=1.5)
    def results_panel() -> None:
        query = st.session_state.get("query", "")
        if not query:
            _card(
                '<div class="t">三步驟，找到你的東西</div>'
                '<div class="step"><span class="num">1</span>'
                '在上方輸入要找的物品（中文也通），或點一下快速標籤。</div>'
                '<div class="step"><span class="num">2</span>'
                '看上方畫面 —— 找到的目標會立刻用綠框鎖定。</div>'
                '<div class="step"><span class="num">3</span>'
                '不在畫面裡？這裡會列出它最後出現的位置與照片；'
                '還沒紀錄時會自動用 AI 找最相似的東西。</div>'
            )
            return

        state = api_get("/state") or {}
        tracks = [t for t in state.get("tracks", [])
                  if query in t["label"] or t["label"] in query]

        if tracks:
            st.markdown(f'<span class="badge green">就在畫面裡！鎖定 {len(tracks)} 個目標</span>',
                        unsafe_allow_html=True)
            _cards_grid([
                f'<div class="t">{esc(t["display"])}</div>'
                f'<div class="s">{esc(loc_zh(t["location"]))} · '
                f'信心 {t["confidence"]:.0%} · 已持續 {t["age"]:.0f} 秒</div>'
                for t in tracks[:6]
            ], tone="hit")
        else:
            st.markdown('<span class="badge gray">目前不在畫面中，持續掃描…</span>',
                        unsafe_allow_html=True)

        # 記憶回溯：以「同一件物品」為單位的最後行蹤＋習慣
        recall = (api_get("/memory/recall", query=query) or {}).get("objects", [])
        if recall:
            _section("", "記憶回溯", "同一件物品的持續身分")
            for obj in recall[:2]:
                latest = obj.get("latest_episode") or {}
                habit = obj.get("top_habit")
                habit_txt = (f'<br>習慣：通常在 {esc(loc_zh(habit["location"]))}'
                             f'（{habit["pct"]:.0%} 的時間）' if habit else "")
                where = esc(loc_zh(latest.get("location") or obj.get("last_location")))
                when = esc(latest.get("end") or obj.get("last_seen") or "")
                dur = latest.get("duration_min")
                dur_txt = f'（停留 {dur:.0f} 分鐘）' if dur else ""
                _card(
                    f'<div class="row">{_thumb_html(obj.get("thumb"))}<div>'
                    f'<div class="t">{esc(obj["display"])}</div>'
                    f'<div class="s">上次：{where} · 🕒 {when}{dur_txt}'
                    f'{habit_txt}</div></div></div>'
                )

        hist = (api_get("/history", query=query, limit=6) or {}).get("records", [])
        if hist:
            _section("🕒", "最後出現位置")
            _cards_grid([
                f'<div class="row">{_thumb_html(r.get("thumb"))}<div>'
                f'<div class="t">{esc(r["display"])}</div>'
                f'<div class="s">{esc(loc_zh(r["location"]))} · 🕒 {esc(r["last_seen"])}<br>'
                f'信心 {(r["confidence"] or 0):.0%}</div></div></div>'
                for r in hist
            ])

        if not tracks and not hist:
            sem = api_get("/semantic", query=query, k=4) or {}
            results = sem.get("results", [])
            if sem.get("mode") == "visual" and results:
                _section("", "AI 找到最相似的", "依視覺語意比對")
                st.caption(f"還沒有「{query}」的直接紀錄，以下是看起來最接近的：")
                _cards_grid([
                    f'<div class="row">{_thumb_html(r.get("thumb"))}<div>'
                    f'<div class="t">{esc(r["display"])} '
                    f'<span class="badge blue">相似 {r["score"]:.0%}</span></div>'
                    f'<div class="s">{esc(loc_zh(r["location"]))} · 🕒 {esc(r["last_seen"])}'
                    f'</div></div></div>'
                    for r in results
                ])
            elif sem.get("suggestions"):
                _section("", "也許你想找")
                sug_cols = st.columns(min(3, len(sem["suggestions"])))
                for si, s_ in enumerate(sem["suggestions"]):
                    with sug_cols[si % len(sug_cols)]:
                        if st.button(s_, key=f"sug_{s_}", width="stretch"):
                            st.session_state["pending_query"] = s_
                            st.rerun(scope="app")
            else:
                _card('<div class="t">還沒有任何紀錄</div>'
                      f'<div class="s">相機正在持續為你掃描「{esc(query)}」。'
                      '只要它出現在鏡頭前，就會自動記下位置與照片。</div>')

    results_panel()

    @st.fragment(run_every=3.0)
    def personal_panel() -> None:
        data = api_get("/items") or {}
        items = data.get("items", [])
        if not items:
            return
        _section("", "我的物品", "到「物品註冊管理」分頁可維護")
        cols = st.columns(min(3, len(items)))
        for i, it in enumerate(items):
            with cols[i % len(cols)]:
                if it.get("last_seen"):
                    status = (
                        f'{esc(loc_zh(it.get("last_location")))} · 🕒 {esc(it["last_seen"])}'
                        f'<br>辨識相似度 {(it.get("last_score") or 0):.0%}'
                    )
                else:
                    status = "尚未在畫面中發現"
                _card(
                    f'<div class="row">{_thumb_html(it.get("thumb"))}<div>'
                    f'<div class="t">{esc(it["name"])}</div>'
                    f'<div class="s">{status}</div></div></div>'
                )
                b1, b2 = st.columns(2)
                if b1.button("找它", key=f"find_{it['item_id']}", width="stretch"):
                    st.session_state["pending_query"] = it["label"]
                    st.rerun(scope="app")
                if b2.button("移除", key=f"rm_{it['item_id']}", width="stretch"):
                    api_delete(f"/items/{it['item_id']}")
                    st.rerun(scope="app")

    personal_panel()

# ------------------------------------------------------- 記憶時間軸分頁

with tab_memory:
    _section("", "記憶時間軸", "事件 → 情節 → 習慣")
    mem_data = fetch_memory_objects()
    mem_objects = mem_data.get("objects", [])
    if not mem_objects:
        _card('<div class="t">還沒有記憶</div>'
              '<div class="s">身分引擎需要先觀察物品幾次，才會建立「同一件物品」的概念。'
              '讓相機多看一會兒，或搜尋一個畫面中的物品。</div>')
    else:
        mem_left, mem_right = st.columns([1, 1.8], gap="large")
        with mem_left:
            choices = {f'{o["display"]}（{o["n_obs"]} 次目擊）': o["object_id"]
                       for o in mem_objects}
            picked = st.selectbox("選擇物品", list(choices.keys()), key="mem_pick")
            obj = next(o for o in mem_objects if o["object_id"] == choices[picked])
            st.caption(f'首次出現：{obj.get("first_seen") or "—"}'
                       f'  \n最後出現：{loc_zh(obj.get("last_location"))} · {obj.get("last_seen") or "—"}')
            timeline = fetch_memory_timeline(choices[picked])
            habits = timeline.get("habits", [])
            if habits:
                st.markdown("**習慣分佈**（依停留時間加權）")
                for h in habits[:4]:
                    st.progress(min(1.0, h["pct"]),
                                text=f'{loc_zh(h["location"])}　{h["pct"]:.0%}')
            else:
                st.caption("觀測累積還不夠，尚未形成習慣結論。")
        with mem_right:
            eps = (timeline or {}).get("episodes", [])
            if eps:
                st.markdown(f"**情節時間軸**（{len(eps)} 段）")
                for ep in eps[:15]:
                    _card(
                        f'<div class="t">{esc(loc_zh(ep["location"]))}</div>'
                        f'<div class="s">🕒 {esc(ep["start"])} → {esc(ep["end"])}'
                        f'（停留 {ep["duration_min"]:.0f} 分鐘）</div>'
                    )
            else:
                st.caption("此物品在保留期內沒有情節紀錄。")

# ------------------------------------------------------- 物品註冊管理分頁

with tab_manage:
    mg_left, mg_right = st.columns([1, 1.5], gap="large")

    with mg_left:
        _section("➕", "註冊新物品", "讓系統認得「就是這一件」")
        if st.session_state.pop("m_flash", None):  # rerun 後補顯示成功訊息
            st.success(st.session_state.pop("m_flash_msg", "已完成"))
        # （成功訊息由 st.success 自帶樣式，不另加圖示）
        _card(
            '<div class="step"><span class="num">1</span>取個名字（中文可），'
            '並填物品的英文類別。</div>'
            '<div class="step"><span class="num">2</span>上傳 3~5 張不同角度的特寫照'
            '（物品佔畫面大半最好）。</div>'
            '<div class="step"><span class="num">3</span>註冊後到右側用「測試比對」'
            '確認系統認得出來。</div>'
        )
        # file_uploader 無法以程式清空，成功後輪替 key 讓整組表單以全新實例重生
        m_rev = st.session_state.setdefault("m_form_rev", 0)
        st.text_input("物品名稱（可中文）", key=f"m_name_{m_rev}", placeholder="我的黑色錢包")
        st.text_input("物品類別（英文）", key=f"m_label_{m_rev}", placeholder="wallet",
                      help="這是 AI 偵測用的類別字，例如 wallet、keys、water bottle")
        m_files = st.file_uploader("參考照片", type=["jpg", "jpeg", "png"],
                                   accept_multiple_files=True, key=f"m_files_{m_rev}")
        if m_files:
            st.image([f.getvalue() for f in m_files], width=88)
        st.caption("照片只取縮圖與視覺特徵，原圖不會儲存。")
        if st.button("註冊物品", width="stretch", key="m_register", type="primary"):
            name = st.session_state[f"m_name_{m_rev}"].strip()
            label = st.session_state[f"m_label_{m_rev}"].strip()
            if not name or not label or not m_files:
                st.warning("請填寫名稱、英文類別，並至少上傳一張照片。")
            elif not label.isascii():
                st.warning("物品類別請使用英文（例如 wallet、keys、water bottle）。")
            else:
                with st.spinner("正在處理照片並計算視覺特徵…"):
                    payload = {"name": name, "label": label,
                               "images": [photo_to_b64(f) for f in m_files]}
                    res, err = api_post_long("/items", payload)
                if res and res.get("item_id"):
                    # st.rerun 會沖掉當下訊息，改用旗標讓訊息在重繪後顯示；
                    # 同時輪替表單 key，名稱/類別/照片全部歸零
                    st.session_state["m_form_rev"] = m_rev + 1
                    st.session_state["m_flash"] = True
                    st.session_state["m_flash_msg"] = (
                        f"已註冊「{name}」（{res['n_photos']} 張參考照），"
                        "右側清單可檢視與測試。"
                    )
                    st.rerun()
                else:
                    st.error(f"註冊失敗：{err}")

    with mg_right:
        _section("", "已註冊物品")
        mg_rev = st.session_state.setdefault("mg_rev", 0)  # 操作成功後輪替，清空各表單
        m_data = api_get("/items") or {}
        m_items = m_data.get("items", [])
        if not m_data.get("available", True):
            st.warning("後端 CLIP 影像引擎未啟用，註冊功能暫不可用。")
        if not m_items:
            st.caption("尚未註冊任何物品 —— 從左側開始第一件吧。")

        for it in m_items:
            iid = it["item_id"]
            with st.expander(f"{it['name']}（{it['label']} · {it['n_photos']} 張參考照）"):
                if it.get("last_seen"):
                    st.caption(f"最後出現：{loc_zh(it.get('last_location'))} · "
                               f"🕒 {it['last_seen']} · 相似度 {(it.get('last_score') or 0):.0%}")
                else:
                    st.caption("尚未在畫面中發現")

                detail = api_get(f"/items/{iid}") or {}
                photos = detail.get("photos", [])
                if photos:
                    st.markdown("**參考照片**")
                    pcols = st.columns(min(5, max(1, len(photos))))
                    for j, ph in enumerate(photos):
                        with pcols[j % len(pcols)]:
                            if ph.get("thumb"):
                                st.image(base64.b64decode(ph["thumb"]),
                                         width="stretch")
                            if len(photos) > 1 and st.button(
                                    "移除", key=f"mp_{iid}_{ph['photo_id']}",
                                    width="stretch"):
                                api_delete(f"/items/{iid}/photos/{ph['photo_id']}")
                                st.rerun()

                add_files = st.file_uploader("補充參考照（拍得越多認得越準）",
                                             type=["jpg", "jpeg", "png"],
                                             accept_multiple_files=True,
                                             key=f"madd_{iid}_{mg_rev}")
                ac1, ac2 = st.columns(2)
                if ac1.button("➕ 加入照片", key=f"maddbtn_{iid}", width="stretch"):
                    if add_files:
                        with st.spinner("計算特徵…"):
                            res, err = api_post_long(
                                f"/items/{iid}/photos",
                                {"images": [photo_to_b64(f) for f in add_files]})
                        if err:
                            st.error(err)
                        else:
                            st.session_state["mg_rev"] = mg_rev + 1  # 清空上傳框
                            st.rerun()
                    else:
                        st.warning("請先選擇照片。")
                if ac2.button("刪除這件物品", key=f"mdel_{iid}", width="stretch"):
                    api_delete(f"/items/{iid}")
                    st.rerun()

                rn1, rn2 = st.columns([3, 1])
                rn1.text_input("重新命名", key=f"mre_{iid}_{mg_rev}", placeholder=it["name"],
                               label_visibility="collapsed")
                if rn2.button("改名", key=f"mrebtn_{iid}", width="stretch"):
                    new_name = st.session_state.get(f"mre_{iid}_{mg_rev}", "").strip()
                    if new_name:
                        api_patch(f"/items/{iid}", {"name": new_name})
                        st.session_state["mg_rev"] = mg_rev + 1  # 清空改名欄
                        st.rerun()

                st.markdown("**測試比對** — 上傳一張現場照，確認系統認不認得出它")
                tf = st.file_uploader("測試照片", type=["jpg", "jpeg", "png"],
                                      key=f"mtest_{iid}", label_visibility="collapsed")
                if tf is not None and st.button("開始測試", key=f"mtestbtn_{iid}",
                                                width="stretch"):
                    with st.spinner("比對中…"):
                        res, err = api_post_long(f"/items/{iid}/test",
                                                 {"image": photo_to_b64(tf)})
                    if err:
                        st.error(err)
                    else:
                        score = res.get("score") or 0.0
                        thr = res.get("threshold", 0.75)
                        # 餘弦相似度理論上可為負，st.progress 只收 0~1
                        st.progress(max(0.0, min(1.0, score)),
                                    text=f"相似度 {score:.0%}（判定門檻 {thr:.0%}）")
                        if res.get("passed"):
                            st.success("認得出來：這張照片會被判定為這件物品。")
                        else:
                            st.warning("認不出來：建議補拍更近、更多角度的參考照；"
                                       "或在 config.yaml 調低 personal_min_score。")

# --------------------------------------------------------------------- footer

st.markdown(f'<div class="foot">CtrlF v{UI_VERSION} · YOLO-World × CLIP · '
            '所有影像皆在本機運算，不上傳雲端</div>', unsafe_allow_html=True)
