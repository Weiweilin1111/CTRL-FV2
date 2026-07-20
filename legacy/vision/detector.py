import cv2
import time
import sys
import os
import math
from ultralytics import YOLO

# 確保能導入 api.database
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.database import add_record

class ObjectDetector:
    TARGET_LABELS = ['pen', 'phone', 'key', 'bottle', 'cup', 'laptop']
    AREA_LABELS = [
        "bed",           # 床
        "desk",          # 書桌
        "table",         # 一般桌子/餐桌
        "nightstand",    # 床頭櫃
        "shelf",         # 層架/書架
        "chair",         # 椅子
        "cabinet",       # 櫃子/抽屜櫃
        "floor",         # 地板
        "sofa",          # 沙發
        "windowsill"     # 窗台
    ]

    def __init__(self, model_path="yolov8x-world.pt"):
        """
        初始化 YOLO-World 模型
        """
        self.model = YOLO(model_path)
        self.user_query_list = []
        
        # 1. 動態掃描清單
        self.scan_registry = set(self.TARGET_LABELS)
        
        # 2. Centroid Tracking
        self.active_objects = {}  
        self.next_instance_ids = {}

    def set_targets(self, targets: list, background_targets: list = None):
        """
        設定模型要偵測的目標類別，包含當下想特別標註畫出來的目標與背景常駐目標。
        """
        if background_targets is None:
            background_targets = []
            
        self.user_query_list = targets.copy()
        self.background_query_list = background_targets.copy()
        
        # 重新初始化動態掃描清單 (包含預設的 TARGET_LABELS 與最新清單)
        self.scan_registry = set(self.TARGET_LABELS)
        
        # 將目標與背景目標都加入掃描清單 (寫進 DB)
        for cat in targets + background_targets:
            if cat.strip():
                self.scan_registry.add(cat.strip().lower())
                
        # 將 scan_registry 與 AREA_LABELS 結合交給 YOLO
        combined_targets = list(self.scan_registry.union(set(self.AREA_LABELS)))
        self.model.set_classes(combined_targets)

    def process(self, frame):
        """
        執行推論、將標準化座標轉換回原圖尺寸，並畫框
        回傳: (annotated_frame, detection_data)
        """
        results = self.model(frame, imgsz=960, conf=0.15, iou=0.45, agnostic_nms=True)
        result = results[0]
        
        h, w = frame.shape[:2]
        annotated_frame = frame.copy()
        detection_data = []
        
        if result.boxes is not None:
            boxes_n = result.boxes.xyxyn.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            # 更新 missing_frames
            for cls_name, ids_dict in self.active_objects.items():
                for inst_id, obj_data in ids_dict.items():
                    obj_data['missing_frames'] += 1
                    
            parsed_objects = []
            dynamic_anchors = []
            
            for box_n, conf, cls_id in zip(boxes_n, confs, class_ids):
                x1 = int(box_n[0] * w)
                y1 = int(box_n[1] * h)
                x2 = int(box_n[2] * w)
                y2 = int(box_n[3] * h)
                
                class_name = result.names[int(cls_id)].lower()
                confidence = float(conf)
                
                if class_name in self.AREA_LABELS:
                    dynamic_anchors.append({
                        "name": class_name,
                        "bbox": (x1, y1, x2, y2)
                    })
                elif confidence < 0.2:
                    continue  # 非 AREA_LABELS 的目標，信心值低於 0.2 則忽略
                
                parsed_objects.append((class_name, confidence, x1, y1, x2, y2))
            
            used_match_ids = {cls: set() for cls in self.active_objects.keys()}
            
            for class_name, confidence, x1, y1, x2, y2 in parsed_objects:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                active_ids = self.active_objects.setdefault(class_name, {})
                min_dist = float('inf')
                matched_id = None
                
                # 尋找最近的現有 object (距離 < 100 像素)
                for inst_id, obj_data in active_ids.items():
                    if inst_id in used_match_ids.get(class_name, set()):
                        continue
                        
                    dist = math.hypot(cx - obj_data['cx'], cy - obj_data['cy'])
                    if dist < 100 and dist < min_dist:
                        min_dist = dist
                        matched_id = inst_id
                
                should_write_db = False
                
                if matched_id is not None:
                    # 更新既有 ID
                    used_match_ids.setdefault(class_name, set()).add(matched_id)
                    obj_data = active_ids[matched_id]
                    obj_data['cx'] = cx
                    obj_data['cy'] = cy
                    obj_data['missing_frames'] = 0
                    
                    # db_cx, db_cy 是上次寫入的座標
                    db_dist = math.hypot(cx - obj_data['db_cx'], cy - obj_data['db_cy'])
                    
                    # 取出上次存入 DB 的時間
                    last_db_time = obj_data.get('last_db_time', 0)
                    current_time = time.time()
                    
                    # 位移超過 50，或者信心值創新高，或者間隔超過 10 秒再拋給 DB (DB 會另外管控 10 分鐘)
                    if db_dist > 50 or confidence > obj_data['max_conf'] or (current_time - last_db_time) > 10:
                        obj_data['db_cx'] = cx
                        obj_data['db_cy'] = cy
                        obj_data['last_db_time'] = current_time
                        if confidence > obj_data['max_conf']:
                            obj_data['max_conf'] = confidence
                        should_write_db = True
                    instance_id = matched_id
                else:
                    # 新物件
                    instance_id = self.next_instance_ids.get(class_name, 1)
                    self.next_instance_ids[class_name] = instance_id + 1
                    used_match_ids.setdefault(class_name, set()).add(instance_id)
                    
                    active_ids[instance_id] = {
                        'cx': cx, 'cy': cy, 'db_cx': cx, 'db_cy': cy,
                        'max_conf': confidence, 'missing_frames': 0,
                        'last_db_time': time.time()
                    }
                    should_write_db = True
                    
                # 寫入資料庫
                if class_name in self.scan_registry:
                    if should_write_db and confidence > 0.25:
                        location = 'floor'
                        min_area = float('inf')
                        for anchor in dynamic_anchors:
                            ax1, ay1, ax2, ay2 = anchor["bbox"]
                            if ax1 <= cx <= ax2 and ay1 <= cy <= ay2:
                                area_size = (ax2 - ax1) * (ay2 - ay1)
                                if area_size < min_area:
                                    min_area = area_size
                                    location = anchor["name"]
                        
                        bbox_coords = (x1, y1, x2, y2)
                        add_record(class_name, instance_id, location, confidence, bbox_coords)
                
                detection_data.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_name": class_name,
                    "instance_id": instance_id
                })
                
                # 畫面上顯示過濾 (display_results)
                # 監視畫面只顯示「手動查詢」的詞彙框框，常駐背景物件不顯示
                should_draw = False
                
                if self.user_query_list:
                    for query in self.user_query_list:
                        if query.strip() and query.strip().lower() in class_name.lower():
                            should_draw = True
                            break

                if should_draw:
                    color = (255, 0, 0) if class_name in self.AREA_LABELS else (0, 255, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # 加上文字標籤與目標追蹤編號 (讓畫面除了框，還有文字呈現)
                    label_text = f"{class_name}-{instance_id}"
                    cv2.putText(annotated_frame, label_text, (x1, max(y1 - 10, 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    label = f"{class_name}_{instance_id} {confidence:.2f}"
                    cv2.putText(
                        annotated_frame, 
                        label, 
                        (x1, max(15, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        color, 
                        2
                    )
            
            # 刪除消失超過一分鐘 (1800 幀，假設以 30FPS 計算) 的物件
            for cls_name, ids_dict in list(self.active_objects.items()):
                keys_to_remove = [k for k, v in ids_dict.items() if v['missing_frames'] >= 1800]
                for k in keys_to_remove:
                    del ids_dict[k]
                if not ids_dict:
                    del self.active_objects[cls_name]

        return annotated_frame, detection_data
