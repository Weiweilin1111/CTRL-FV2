import sqlite3
import os
import time
import uuid
import math
import json
from datetime import datetime, timedelta
import threading
import queue
import chromadb
from chromadb.utils import embedding_functions

# 確保路徑指向 api 資料夾下的 detections.db
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detections.db')
CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chroma_db')

_write_queue = queue.Queue()
_db_worker_started = False

# ChromaDB 初始化
chroma_client = None
chroma_collection = None
emb_fn = None

def init_chroma():
    global chroma_client, chroma_collection, emb_fn
    if chroma_client is None:
        # 使用 PersistentClient 將向量資料存到硬碟
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        # 指定 all-MiniLM-L6-v2 模型
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        chroma_collection = chroma_client.get_or_create_collection(
            name="object_embeddings",
            embedding_function=emb_fn
        )
        print(f"✅ ChromaDB 已初始化於: {CHROMA_PATH}")
        
        # [架構優化] 詞彙預熱 (Pre-population)
        # 將預設你想追蹤的物件(例如 TARGET_LABELS 中的項目)先建入向量的標準詞庫中，
        # 作為向量搜尋的「錨點」。這可以統一使用者的同義詞搜尋(例如 mobile -> phone)
        predefined_labels = ['pen', 'phone', 'key', 'bottle', 'cup', 'laptop', 'book', 'wallet', 'bag']
        for label in predefined_labels:
            try:
                chroma_collection.upsert(
                    ids=[f"predefined_anchor_{label}"],
                    documents=[label],
                    metadatas=[{"name": label, "location": "predefined", "confidence": 1.0, "last_seen": "1970-01-01 00:00:00"}]
                )
            except Exception as e:
                pass
        print("✅ 預設系統標準詞庫 (Semantic Anchors) 已寫入 ChromaDB")

def get_connection():
    return sqlite3.connect(DB_PATH)

_db_initialized = False

def init_db():
    global _db_worker_started, _db_initialized
    if _db_initialized:
        return
        
    with get_connection() as conn:
        cursor = conn.cursor()
        # 1. Schema 規格：確保 items 表中的 item_id 設為 UNIQUE 或 PRIMARY KEY
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS items (
                item_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                location TEXT, 
                bbox_coordinates TEXT,
                confidence REAL,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # 常駐清單
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fixed_list (
                tag TEXT PRIMARY KEY
            )
        ''')
        # 近期搜尋清單
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recent_list (
                tag TEXT PRIMARY KEY,
                last_used DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
    
    # 初始化 ChromaDB
    init_chroma()
    
    # 啟動時自動清理
    auto_clean_db(days=7)
    
    # Blackwell 優化：背景非同步寫入，確保在高 FPS 偵測下不產生延遲
    if not _db_worker_started:
        threading.Thread(target=_db_worker_loop, daemon=True).start()
        _db_worker_started = True

    print(f"✅ 資料庫已初始化於: {DB_PATH}")
    _db_initialized = True

def auto_clean_db(days=7):
    # TTL 自動清理：刪除 last_seen 超過指定天數的舊紀錄
    time_limit = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cursor = conn.cursor()
        # 找出要刪除的 ID 以便同步刪除 ChromaDB
        cursor.execute('SELECT item_id FROM items WHERE last_seen < ?', (time_limit,))
        old_records = cursor.fetchall()
        ids_to_delete = [r[0] for r in old_records]
        
        if ids_to_delete:
            cursor.execute('DELETE FROM items WHERE last_seen < ?', (time_limit,))
            conn.commit()
            
            # 同步從 ChromaDB 移除對應的紀錄
            try:
                chroma_collection.delete(ids=ids_to_delete)
                print(f"🧹 已清理 {len(ids_to_delete)} 筆超過 7 天的舊紀錄 (SQLite & ChromaDB)")
            except Exception as e:
                print(f"ChromaDB delete error: {e}")

def _db_worker_loop():
    while True:
        try:
            task = _write_queue.get()
            if task is None:
                break
            
            # Unpacking variables
            item_uid, name, location, conf, bbox = task
            current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            bbox_str = json.dumps(bbox) if bbox else "[]"
            
            # 計算當前物件的幾何中心點 (Cx, Cy)
            new_cx, new_cy = 0, 0
            if bbox and len(bbox) == 4:
                new_cx = (bbox[0] + bbox[2]) / 2
                new_cy = (bbox[1] + bbox[3]) / 2

            with get_connection() as conn:
                cursor = conn.cursor()
                
                # 取出該物件的最新紀錄，以進行時間與空間比對
                cursor.execute('SELECT item_id, location, last_seen, bbox_coordinates FROM items WHERE name = ? ORDER BY last_seen DESC LIMIT 1', (name,))
                row = cursor.fetchone()
                
                should_insert = True
                target_id = item_uid  # 如果是新物件，使用新分配的 UUID
                
                if row:
                    prev_id, prev_location, last_seen_str, prev_bbox_str = row
                    try:
                        last_seen_dt = datetime.strptime(last_seen_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        last_seen_dt = datetime.now()
                        
                    # 解析舊紀錄的中心點
                    prev_bbox = json.loads(prev_bbox_str) if prev_bbox_str else []
                    prev_cx, prev_cy = 0, 0
                    if prev_bbox and len(prev_bbox) == 4:
                        prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2
                        prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2
                        
                    # 計算歐幾里德距離與時間差
                    distance = math.sqrt((new_cx - prev_cx)**2 + (new_cy - prev_cy)**2)
                    time_diff = (datetime.now() - last_seen_dt).total_seconds()
                    
                    # 條件式更新：位移 <= 50px，相同語義位置，且時間差 <= 600秒 (10分鐘)
                    if distance <= 50 and prev_location == location and time_diff <= 600:
                        should_insert = False
                        target_id = prev_id
                
                # Try / Atomic Write 核心防護區塊
                try:
                    cursor.execute('BEGIN')
                    
                    if should_insert:
                        cursor.execute('''
                            INSERT INTO items (item_id, name, location, bbox_coordinates, confidence, last_seen)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (target_id, name, location, bbox_str, conf, current_time_str))
                    else:
                        cursor.execute('''
                            UPDATE items SET 
                                location=?, bbox_coordinates=?, confidence=?, last_seen=?
                            WHERE item_id=?
                        ''', (location, bbox_str, conf, current_time_str, target_id))
                    
                    # 嘗試寫入 ChromaDB
                    chroma_collection.upsert(
                        ids=[target_id],
                        documents=[name], 
                        metadatas=[{"name": name, "location": location, "confidence": conf, "last_seen": current_time_str}]
                    )
                    
                    # 若兩邊都寫入成功，正式 Commit
                    conn.commit()
                    
                except Exception as sync_e:
                    # 階段二若異常，執行 Rollback 確保無孤兒紀錄
                    conn.rollback()
                    print(f"❌ [雙軌防護] 異質資料庫同步發生異常，交易已撤銷 (Rollback): {sync_e}")

            _write_queue.task_done()
        except Exception as e:
            print(f"DB worker loop error: {e}")

def add_record(name: str, instance_id: int, area: str, conf: float, bbox: tuple):
    item_uid = str(uuid.uuid4())
    display_name = f"{name}-{instance_id}" if instance_id is not None else name
    _write_queue.put((item_uid, display_name, area, conf, bbox))

def get_semantic_closest(query: str):
    """
    純粹計算並回傳語義最接近的詞彙與距離
    """
    if not query:
        return query, 0.0
            
    try:
        global chroma_collection
        if chroma_collection is None:
            init_chroma()
            
        if chroma_collection is not None:
            results = chroma_collection.query(
                query_texts=[query],
                n_results=1
            )
            distances = results.get('distances', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            
            if distances:
                closest_tag = metadatas[0].get('name')
                dist = distances[0]
                return closest_tag, dist
    except Exception as e:
        print(f"ChromaDB search error: {e}")
        
    return query, 0.0

def get_history(name: str):
    # 用確定的標籤去歷史紀錄找
    with get_connection() as conn:
        cursor = conn.cursor()
        # 透過 GROUP BY name 與 max(last_seen)，確保同一編號 (例如 pen-1, pen-2) 只顯示最新的一筆
        cursor.execute('''
            SELECT item_id, name, location, confidence, max(last_seen) as last_seen FROM items
            WHERE name LIKE ?
            GROUP BY name
            ORDER BY last_seen DESC
            LIMIT 5
        ''', (f"{name}%",))
        records = cursor.fetchall()
    
    return [
        {"item_id": r[0], "name": r[1], "area": r[2], "conf": r[3], "timestamp": r[4]}
        for r in records
    ]

def add_recent_tag(tag: str):
    # 近期搜尋清單：自動存入新標籤
    tag = tag.strip().lower() # 全英文處理
    if not tag:
        return
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO recent_list (tag, last_used)
            VALUES (?, ?)
            ON CONFLICT(tag) DO UPDATE SET last_used=excluded.last_used
        ''', (tag, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        # 上限 10 個
        cursor.execute('''
            DELETE FROM recent_list
            WHERE tag NOT IN (
                SELECT tag FROM recent_list ORDER BY last_used DESC LIMIT 10
            )
        ''')
        conn.commit()

def get_recent_tags():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT tag FROM recent_list ORDER BY last_used DESC')
        return [r[0] for r in cursor.fetchall()]

def get_fixed_tags():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT tag FROM fixed_list')
        return [r[0] for r in cursor.fetchall()]

def add_fixed_tag(tag: str):
    tag = tag.strip().lower()
    if tag:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT OR IGNORE INTO fixed_list (tag) VALUES (?)', (tag,))
            conn.commit()
            
def remove_fixed_tag(tag: str):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM fixed_list WHERE tag = ?', (tag,))
        conn.commit()