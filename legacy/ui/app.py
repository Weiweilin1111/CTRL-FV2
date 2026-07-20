import streamlit as st
import streamlit.components.v1 as components
import requests
import time

API_BASE = "http://localhost:8000"

def set_search_query(query: str):
    if query:
        requests.post(f"{API_BASE}/recent_tags", json={"tag": query})
    try:
        requests.get(f"{API_BASE}/search_get", params={"query": query})
    except requests.exceptions.RequestException as e:
        st.error(f"無法連接到後端 API: {e}")

# 當搜尋框內容改變時觸發的回呼函數
def update_search():
    query = st.session_state.get("search_input", "").strip()
    set_search_query(query)

# 設定網頁標題
st.set_page_config(page_title="CtrlF - 即時尋物系統", page_icon="🔍")

st.title("CtrlF - 即時尋物系統 ")
st.write("請輸入你想尋找的物品名稱，系統將自動標記出符合的目標。")
with st.sidebar:
    st.header("歷史紀錄")
    history_query = st.text_input("輸入欲查詢的歷史物件:", key="history_input")
    if st.button("尋找歷史位置"):
        if history_query:
            try:
                records = get_history(history_query)
                if records:
                    for r in records:
                        # name_display 改回使用 name (因為 item_id 現在是 UUID)
                        name_display = r['name']
                        st.info(f"📍 **{name_display}** 在 **{r['area']}** 被偵測到\n\n🕒 時間: {r['timestamp']}\n\n信心度: {r['conf']:.2f}")
                else:
                    st.warning(f"找不到關於 '{history_query}' 的紀錄。")
            except Exception as e:
                st.error(f"查詢失敗: {e}")
        else:
            st.warning("請先輸入要查詢的物件名稱。")
            
    st.divider()
    st.header("常駐清單管理")
    new_fixed_tag = st.text_input("新增常駐物件標籤:")
    if st.button("加入常駐"):
        if new_fixed_tag:
            requests.post(f"{API_BASE}/fixed_tags", json={"tag": new_fixed_tag})
            update_search()  # 立即同步至後端
            st.success(f"已加入常駐: {new_fixed_tag}")
            st.rerun()
            
    fixed_tags = []
    try:
        res = requests.get(f"{API_BASE}/fixed_tags")
        if res.status_code == 200:
            fixed_tags = res.json().get("tags", [])
    except:
        pass
        
    if fixed_tags:
        st.write("已設定常駐物件:")
        for tag in fixed_tags:
            col1, col2 = st.columns([3, 1])
            col1.write(f"- {tag}")
            if col2.button("刪除", key=f"del_fixed_{tag}"):
                requests.delete(f"{API_BASE}/fixed_tags/{tag}")
                update_search()  # 立即同步至後端
                st.rerun()

def update_search_from_select():
    selected_option = st.session_state.get("select_input", "(無)")
    if selected_option != "(無)":
        clean_tag = selected_option.replace("🌟 ", "").replace("🕒 ", "")
        st.session_state.search_input = clean_tag
        update_search()

def update_search_with_tag(tag):
    st.session_state.search_input = tag
    update_search()

# 獲取最近搜尋選單項目
recent_tags = []
try:
    res = requests.get(f"{API_BASE}/recent_tags")
    if res.status_code == 200:
        recent_tags = res.json().get("tags", [])
except:
    pass

# 提供選單讓使用者選取常駐或近期物件，或是手動輸入
st.subheader("快速選擇搜尋或手動輸入")
col_sel, col_inp = st.columns(2)

with col_sel:
    all_options = ["(無)"] + [f"🌟 {t}" for t in fixed_tags] + [f"🕒 {t}" for t in recent_tags]
    st.selectbox("從清單選擇:", all_options, key="select_input", on_change=update_search_from_select)
        
with col_inp:
    # 搜尋框
    st.text_input("輸入欲尋找的物件 (例如: cup, person, cell phone):", 
                  key="search_input", 
                  on_change=update_search)

# 若尋找物件時，順便在下方提供相應的歷史紀錄（解決查不到時可以直接看紀錄）
current_query = st.session_state.get("search_input", "").strip()

if current_query:
    records = []
    try:
        res = requests.get(f"{API_BASE}/history", params={"query": current_query})
        if res.status_code == 200:
            records = res.json().get("records", [])
    except:
        pass
    
    if records:
        # 第一種狀況：精確紀錄找到了
        st.success(f"🎯 現場已捕捉到/曾存在 '{current_query}' 的紀錄！")
        with st.expander(f"🔍 點此查看 '{current_query}' 的最後出現位置", expanded=True):
            cols = st.columns(min(len(records), 5))
            for i, r in enumerate(records[:5]):
                with cols[i]:
                    st.info(f"📍 **{r['area']}**\n\n🕒 {r['timestamp'].split(' ')[1]}\n\n{r['name']}")
            
            if st.button("🔄 刷新最新狀態", use_container_width=True):
                pass
    else:
        # 第二種狀況：歷史與現場目前都還沒有「確切字詞」的紀錄
        closest_tag, dist = "", 0.0
        try:
            res = requests.get(f"{API_BASE}/semantic_closest", params={"query": current_query})
            if res.status_code == 200:
                data = res.json()
                closest_tag = data.get("closest_tag")
                dist = data.get("distance")
        except:
            pass
        
        # 設定一個適當的距離門檻 (像是 1.25) 判斷是否為同義關聯詞
        if closest_tag and closest_tag != current_query and dist <= 1.35:
            st.warning(f"目前相機尚未捕捉到 '{current_query}'。已自動為您尋找最關聯的庫存同義對象：'{closest_tag}' (語意距離: {dist:.2f})")
            
            fallback_records = []
            try:
                res = requests.get(f"{API_BASE}/history", params={"query": closest_tag})
                if res.status_code == 200:
                    fallback_records = res.json().get("records", [])
            except:
                pass

            if fallback_records:
                with st.expander(f"📁 查看關聯物件 '{closest_tag}' 的歷史紀錄", expanded=True):
                    cols = st.columns(min(len(fallback_records), 5))
                    for i, r in enumerate(fallback_records[:5]):
                        with cols[i]:
                            st.info(f"📍 **{r['area']}**\n\n🕒 {r['timestamp'].split(' ')[1]}\n\n{r['name']}")
            else:
                st.info(f"關聯物件 '{closest_tag}' 目前也尚未有出現紀錄...")
                
            col1, col2 = st.columns(2)
            with col1:
                if st.button("刷新捕捉狀態", use_container_width=True):
                    pass
            with col2:
                st.button(
                    f"將 '{closest_tag}' 送給相機捕捉", 
                    use_container_width=True, 
                    on_click=update_search_with_tag, 
                    args=(closest_tag,)
                )
        else:
            # 第三種狀況：沒有確切紀錄，也沒有夠近的關聯物件
            st.warning(f"📝 庫存中無 '{current_query}' 或其他高度相關物件的出現紀錄。(相機已正在為您尋找 '{current_query}'，請稍候並刷新！)")
            if st.button("🔄 刷新捕捉狀態 (獲取最新結果)", use_container_width=True):
                pass

st.divider()

st.subheader("即時攝影機畫面")

# 關鍵：使用 HTML img 標籤嵌入串流，讓 HTML 自動處理串流刷新，不使用 st.rerun()
# 避免影像串流跟 UI 之間不斷互相刷新，我們移除時間戳快取干擾
html_code = """
<div style="display: flex; justify-content: center; align-items: center; width: 100%;">
    <img src="http://localhost:8000/video_feed" 
         style="max-width: 100%; border: 2px solid #666; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.5);" 
         alt="Video Feed">
</div>
"""

components.html(html_code, height=600)
