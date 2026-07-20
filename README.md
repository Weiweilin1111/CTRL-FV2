# CtrlF — 即時智慧尋物系統

> 對著鏡頭問「我的鑰匙呢？」——畫面即時鎖定、查最後出現位置、認得「就是你那一件」。
>
> **v3.2.1** · YOLO-World（開放詞彙偵測）× CLIP（跨模態視覺語意）二階段架構
> FastAPI 非同步後端 · Producer-Consumer 多執行緒管線 · Streamlit 前端 · 全程本機運算，影像不上雲
>
> 研究定位：**CtrlF-Memory — 基於視覺記憶建構之智慧環境搜尋助理**
> 記憶建構鏈：Frame → Object（持續身分）→ Episode（情節）→ Habit（習慣）

---

## 功能特色

| 功能 | 說明 |
|------|------|
| **中文搜尋** | 直接輸入「鑰匙」「我的錢包在哪」，內建約 90 詞辭典＋自訂同義詞，自動翻譯成偵測詞彙 |
| **即時鎖定** | 搜尋目標在畫面中以綠框＋半透明高亮即時標記，串流 24fps 不受推論速度影響 |
| **最後出現位置** | 每件物品的位置、時間、**實物縮圖**自動入庫；找不到時直接看它最後在哪 |
| **語意搜尋** | 查無直接紀錄時，自動用 CLIP 跨模態比對：搜 "mobile" 也能命中 cell phone 的目擊照片 |
| **視覺記憶** | 持續身分引擎跨越重啟與消失：`mug#1` 永遠是同一個馬克杯；情節時間軸（09:12 書桌 → 14:30 沙發）與習慣統計（「通常在書桌，70%」）自動建構 |
| **個人物品註冊** | 上傳 3~5 張照片，系統認得「就是這一件」：我的錢包 vs 任何一個錢包 |
| **註冊管理** | 照片庫逐張管理、補照、改名、**測試比對工具**（上傳現場照立即回報認不認得出） |
| **動態調校** | 信心門檻、畫面變化敏感度、AI 分析頻率——側欄滑桿即時生效，不需重啟 |
| **資源友善** | 動態閘控（畫面靜止不推論）＋ FP16 ＋ ROI 遮罩，VRAM 約 1GB；紀錄 7 天 TTL 自動清理 |

## 快速啟動

```powershell
start.bat                                    # 推薦：雙擊或執行皆可，繞過 PowerShell 執行原則
powershell -ExecutionPolicy Bypass -File .\start.ps1   # 或用 PowerShell 版（免改系統設定）
```

> 若直接 `.\start.ps1` 出現「因為這個系統上已停用指令碼執行」——這是 Windows 預設擋未簽署腳本，
> 改用上方 `start.bat`，或先執行一次 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 永久放行使用者腳本。

或分開啟動：

```powershell
python run_api.py          # 後端 http://127.0.0.1:8000（/docs 有互動式 API 文件）
streamlit run ui/app.py    # 前端 http://localhost:8501
```

**首次啟動須知**

- 偵測模型 `yolov8m-worldv2.pt`（約 55MB）會自動下載；離線環境自動退回本地較大的 `yolov8x-world.pt`
- CLIP ViT-B/32 權重與 YOLO-World 共用快取（`~/.cache/clip`），**零額外下載**
- 環境需求：Python 3.10+、依賴見 `requirements.txt`（torch/CUDA 依硬體另裝）

## 系統架構

完整架構圖見 [`architecture.drawio`](architecture.drawio)（可用 [app.diagrams.net](https://app.diagrams.net) 或 VS Code Draw.io 擴充開啟編輯）。

```
                ┌──────────────────────────────────────────────────────────┐
                │                  StateHub（執行期狀態中樞）                │
                │   最新影格信箱 · 軌跡快照 · 效能統計 · JPEG 編碼快取        │
                └──────────────────────────────────────────────────────────┘
                     ▲ 發佈影格              ▲ 發佈結果            │ 只讀快照
┌──────────────┐     │      ┌───────────────┴────────────┐        ▼
│CaptureService │────┘      │     InferencePipeline      │   ┌──────────────────┐
│ (擷取執行緒)  │            │ ① 推論頻率硬上限           │   │ FastAPI (async)  │
│ 相機原生FPS   │            │ ② RoiMask＋像素差分閘控    │   │ /stream /ws      │
│ 斷線自動重連  │            │ ③ YOLO-World-M FP16 @800   │   │ /search /semantic│
└──────────────┘            │ ④ IouTracker 生命週期追蹤  │   │ /items /aliases  │
                            └──────┬──────────────┬──────┘   │ /system/*        │
                                   │ SightingEvent│ EmbedJob │ /history /tags   │
                                   ▼              ▼          └────────┬─────────┘
                          ┌──────────────┐  ┌───────────────┐         │ REST/WS
                          │StorageWriter │  │EmbeddingWorker │        ▼
                          │ 批次寫入     │  │ CLIP 影像嵌入  │  ┌──────────────┐
                          │ SQLite (WAL) │  │ ＋個人物品比對 │  │ Streamlit UI │
                          │ TTL 清理     │  │ → 向量索引 ×2 │  │ fragment局部 │
                          └──────────────┘  └───────────────┘  │ 更新＋骨架屏 │
                                                               └──────────────┘
```

### 核心設計決策

1. **二階段解耦**：YOLO-World 負責「框在哪」（開放詞彙，保住 pen/keys 等非 COCO 類別）；CLIP 對裁切區域計算**影像嵌入**。查詢時文字→CLIP 文字向量→跨模態餘弦搜尋，同義詞問題由視覺語意天然解決。
2. **影像降載三道閘門**：推論頻率硬上限（零成本跳過）→ ROI 遮罩（背景抹黑）→ 像素差分動態觸發（靜止畫面 0 推論，定時抽樣兜底）。
3. **串流與推論徹底脫鉤**：MJPEG 端點只讀 StateHub 快照並在最新影格上合成標註；推論 5fps 時畫面照樣 24fps；N 個分頁共用一份推論與一份 JPEG 編碼。
4. **內嵌向量索引**：尋物場景 <10 萬向量，正規化矩陣內積（精確暴力搜尋）亞毫秒完成，零部署。介面遵循 `VectorStore` Protocol，將來要上 Qdrant/Milvus 寫一個適配器即可。
5. **依賴注入**：`container.py` 是唯一的組裝根；模組之間只依賴 Protocol 契約（`core/interfaces.py`），偵測器/嵌入器/向量庫均可單獨替換。

## 目錄結構

```
ctrlf_v2/
├── config.yaml            # 唯一設定來源（含詳細註解）
├── run_api.py             # 後端入口
├── start.ps1              # 一鍵啟動
├── architecture.drawio    # 系統架構圖（draw.io）
├── ctrlf/
│   ├── config.py          # Pydantic 設定模型
│   ├── container.py       # DI 組裝根（要換實作改這裡）
│   ├── core/              # models.py 資料結構 · interfaces.py Protocol 契約
│   ├── capture/           # camera.py 擷取執行緒 · motion.py ROI＋動態閘控
│   ├── vision/            # detector.py · tracker.py · embedder.py(CLIP)
│   │                      # annotator.py(純顯示) · pipeline.py(消費者執行緒)
│   ├── storage/           # sqlite_repo.py(WAL) · vector_store.py · writer.py
│   ├── runtime/           # hub.py 狀態中樞 · settings.py 熱調參 · vocabulary.py 詞彙
│   │                      # lexicon.py 中文辭典 · personal.py 個人物品
│   │                      # identity.py 持續身分引擎 · memory.py 情節/習慣建構
│   └── api/               # app.py 工廠 · routes_stream/data/personal/memory/system.py
├── ui/app.py              # Streamlit 前端（純消費者，零業務邏輯）
├── tools/                 # rebuild_vectors.py 向量索引重建（由 SQLite 縮圖重算）
├── tests/                 # 常駐測試套件
├── docs/                  # 系統需求書／企劃書與架構圖
└── data/                  # ctrlf.db ＋ 向量索引（自動生成，TTL 自動清理）
```

## 設定重點（config.yaml）

| 區塊 | 關鍵欄位 | 說明 |
|------|---------|------|
| `detector` | `model_candidates` | 依序嘗試載入。效能分級：s@640（eco）/ **m@800（預設）**/ x@960（quality） |
| `detector` | `base_labels` | **開放詞彙模型只偵測詞彙表內的物品**。找不到的東西先加進這裡、常駐清單，或直接搜尋（即時加入） |
| `motion` | `max_infer_fps` / `force_interval` | 推論頻率上限；畫面全靜止時的定時抽樣間隔 |
| `roi` | `enabled` / `rect` | 只偵測指定矩形內，框外背景直接抹黑 |
| `embedder` | `personal_min_score` | 個人物品比對門檻（誤認多→調高；認不出→調低） |
| `storage` | `retention_days` | 目擊紀錄保留天數，逾期自動從 SQLite＋向量索引清除 |

## API 總覽

完整互動式文件：啟動後開 `http://127.0.0.1:8000/docs`

| 端點 | 用途 |
|------|------|
| `GET /stream/mjpeg` · `WS /ws/state` | 即時影像串流；軌跡＋統計推播 |
| `POST /search` | 設定搜尋目標（含中文辭典解析，回傳翻譯結果） |
| `GET /state` · `GET /history` · `GET /semantic` | 當前軌跡；歷史目擊（含縮圖）；CLIP 語意搜尋 |
| `GET /tags` · `POST/DELETE /tags/fixed/…` | 常駐標籤管理 |
| `POST/GET/PATCH/DELETE /items…` | 個人物品：註冊、明細、照片增刪、改名、刪除 |
| `POST /items/{id}/test` | 測試比對：上傳現場照回報相似度與判定 |
| `GET/POST/DELETE /aliases…` | 中文同義詞管理 |
| `GET /memory/objects` · `/memory/timeline` · `/memory/recall` | 視覺記憶：身分總覽、情節時間軸＋習慣、回溯查詢（吃中文） |
| `GET /system/stats` · `GET/POST /system/settings` | 監控（含版本握手）；執行期熱調參 |

## 測試

後端啟動中時，於專案根目錄執行：

```powershell
python -X utf8 tests\test_memory_unit.py   # 記憶層離線單元測試（身分/情節/習慣/TTL 清理，8 項，不需後端）
python -X utf8 tests\test_backend_api.py   # 33 項全端點回歸（搜尋/辭典/物品/記憶/串流/WS…）
python -X utf8 tests\test_ui_apptest.py    # Streamlit AppTest：真實執行 UI 並模擬互動（6 項）
```

UI 有任何改動，務必跑一次 AppTest——它是唯一能抓到頁面執行期錯誤的測試。

## 效能實測（RTX 5070 Ti Laptop）

| 指標 | 實測數值 |
|------|--------|
| 串流 FPS | **30**（與推論頻率完全脫鉤） |
| 推論 | 動態觸發 ~5fps · 43ms/幀（m@800 FP16） |
| 多分頁觀看 | 永遠 1 份推論、1 份 JPEG 編碼（不隨分頁數增加） |
| VRAM | 約 0.7~1.1 GB |
| 同房間辨識數 | 23 件 |
| 儲存增長 | 約 1MB／15 分鐘，7 天 TTL 封頂 |

## 疑難排解

- **`.\start.ps1` 報「已停用指令碼執行」**：Windows 執行原則擋未簽署腳本。改用 `start.bat`，或執行 `powershell -ExecutionPolicy Bypass -File .\start.ps1`，或一次性 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。
- **側欄出現「後端版本不一致」警告**：後端還在跑舊程式碼，關閉後重跑 `start.bat`。
- **啟動失敗：8000 埠被佔用**：舊的 python 程序還活著，工作管理員結束它（或 `Get-NetTCPConnection -LocalPort 8000` 找 PID）。
- **相機打不開**：被其他程式佔用，或 `config.yaml` 的 `camera.index` 不對（外接鏡頭常是 1）。
- **某物品永遠偵測不到**：開放詞彙模型只認詞彙表內的詞——先搜尋一次（會即時加入詞彙）、加入常駐清單，或補進 `base_labels`。側欄打開「顯示所有偵測物件」可看 AI 眼中的全部。
- **中文詞不認識**：搜尋框下方會提示，到側欄「中文同義詞辭典」新增對應即可。
- **個人物品認不出**：用管理分頁的「測試比對」診斷；補拍更近、更多角度的特寫，或調低 `personal_min_score`。
- **向量索引損毀／換了 CLIP 模型**：SQLite 是事實來源，索引可隨時重建——停止後端後執行 `python -X utf8 tools\rebuild_vectors.py`。

## 擴充指南

- **換偵測模型**：改 `model_candidates`（任何 ultralytics World 模型），或實作 `Detector` Protocol 接其他架構（如 YOLOE）。
- **上專業向量庫**：實作 `VectorStore` Protocol 五個方法，在 `container.py` 換掉 `NumpyVectorStore`。
- **多相機**：每路一組 `CaptureService + InferencePipeline`，共用儲存層，`StateHub` 加 camera_id 維度。
- **規劃中**：主動通知（規則引擎＋Windows toast/Telegram）、出門模式（一鍵檢查所有個人物品）、加強搜尋模式、LLM 記憶問答（自然語言查詢情節/習慣）。

> **商業化注意**：ultralytics 為 AGPL-3.0 授權，商用需企業授權或更換偵測器；CLIP 為 MIT。

---

CtrlF v3.2.1 · 所有影像皆在本機運算，不上傳雲端
