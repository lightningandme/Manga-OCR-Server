import cv2
import numpy as np
import os


class MangaCropEngine:
    def __init__(self, easyocr_reader=None):
        self.reader = easyocr_reader

    def get_smart_crop(self, image_bytes, click_x_rel, click_y_rel):
        """
        核心裁剪逻辑：
        1. 自动纠偏点击点
        2. Mode 1: 尝试 OpenCV 几何气泡识别 (最快，最准)
        3. Mode 2: 尝试 EasyOCR 语义聚类 (处理无框/散字)
        4. Mode 3: 动态比例保底 (最后手段)
        """
        # --- 0. 图像解码与预处理 ---
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print("❌ [CropEngine] Image decode failed.")
            return None
        h, w = img.shape[:2]

        # 坐标标准化 (兼容相对坐标与绝对坐标)
        cx = int(click_x_rel * w) if 0 < click_x_rel < 1 else int(click_x_rel)
        cy = int(click_y_rel * h) if 0 < click_y_rel < 1 else int(click_y_rel)

        # 客户端手动全图模式
        if cx == 0 and cy == 0:
            return img

        # --- 1. 自动纠偏 (Search Radius 20px) ---
        # 如果点在空白处，自动吸附到附近的高亮像素(文字/气泡中心)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        search_radius = 20
        min_y, max_y = max(0, cy - search_radius), min(h, cy + search_radius)
        min_x, max_x = max(0, cx - search_radius), min(w, cx + search_radius)
        sub = gray[min_y:max_y, min_x:max_x]

        # 使用高斯模糊找最亮区域，避免噪点干扰
        blurred_sub = cv2.GaussianBlur(sub, (5, 5), 0)
        if blurred_sub.size > 0:
            _, max_val, _, max_loc = cv2.minMaxLoc(blurred_sub)
            if max_val > 180:  # 只有足够亮才纠偏
                cx, cy = min_x + max_loc[0], min_y + max_loc[1]

        # --- 2. 气泡探测 (用于 Mode 1 判断) ---
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        flood_filled = gray.copy()
        # 宽容度设为 18，适应黑白漫画的纸张噪点
        cv2.floodFill(flood_filled, ff_mask, (cx, cy), 255, (18,), (18,), cv2.FLOODFILL_FIXED_RANGE)
        bubble_mask = ff_mask[1:-1, 1:-1] * 255
        cnts, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        is_leaking = True
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            rect_area = bw * bh
            solidity = area / float(rect_area) if rect_area > 0 else 0

            # 漏气判定逻辑：
            # 1. 尺寸过大 (占屏 >80%)
            # 2. 面积过大 (占屏 >60%)
            # 3. 形状过实 (Solidity > 0.9 且面积不小，通常是背景色块而非气泡)
            if not (bw > w * 0.8 or bh > h * 0.8 or area > (w * h * 0.6) or (solidity > 0.9 and area > (w * h * 0.3))):
                is_leaking = False

        # --- 3. 策略分流 ---

        # === Mode 1: 几何气泡模式 ===
        if not is_leaking:
            print(f"🎯 [Mode 1] Bubble Capture triggered at ({cx}, {cy})")
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
            closed_mask = cv2.morphologyEx(bubble_mask, cv2.MORPH_CLOSE, kernel)
            cnts_closed, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if cnts_closed:
                c = max(cnts_closed, key=cv2.contourArea)
                x, y, rw, rh = cv2.boundingRect(c)
                return self._save_debug_and_return(img, x - 10, y - 10, x + rw + 10, y + rh + 10, cx, cy,
                                                   "mode1_bubble")

        # === Mode 2: EasyOCR 语义聚合模式 ===
        print(f"🧠 [Mode 2] Switching to EasyOCR Aggregation at ({cx}, {cy})")
        easy_res = self._try_easyocr_logic(img, gray, cx, cy)
        if easy_res is not None:
            return easy_res

        # === Mode 3: 动态比例保底模式 ===
        print(f"🩹 [Mode 3] Fallback to Proportional Crop at ({cx}, {cy})")
        fw, fh = int(w * 0.6), int(h * 0.8)  # 宽 60%，高 80%
        # 计算中心点，并限制在图像边界内
        x1 = max(0, min(w - fw, cx - fw // 2))
        y1 = max(0, min(h - fh, cy - fh // 2))
        return self._save_debug_and_return(img, x1, y1, x1 + fw, y1 + fh, cx, cy, "mode3_fallback")

    def _try_easyocr_logic(self, img, gray, cx, cy):
        """
        Mode 2 核心：利用 OCR 定位散落的文字，根据距离和 Canny 边缘进行聚类。
        """
        if not self.reader:
            print("⚠️ [Mode 2] EasyOCR reader not initialized.")
            return None

        # 1. 边缘检测 (物理墙)
        # 阈值 (70, 200) 用于忽略细微网点，只保留明显的分镜线和气泡框
        edges = cv2.Canny(gray, 70, 200)

        # 2. OCR 探测
        # text_threshold=0.3 降低门槛，确保能抓到拟声词或模糊字
        try:
            horizontal_list, _ = self.reader.detect(img, text_threshold=0.3)
            raw_boxes = horizontal_list[0] if horizontal_list else []
        except Exception as e:
            print(f"⚠️ [Mode 2] OCR detection failed: {e}")
            return None

        if not raw_boxes:
            print("⚠️ [Mode 2] No text detected.")
            return None

        # 3. 格式标准化 [x1, x2, y1, y2]
        formatted_boxes = []
        for b in raw_boxes:
            if len(b) == 4:  # 标准格式
                formatted_boxes.append(b)
            elif len(b) == 2 and len(b[0]) == 2:  # 多点格式 [[x,y]...]
                xs = [p[0] for p in b];
                ys = [p[1] for p in b]
                formatted_boxes.append([min(xs), max(xs), min(ys), max(ys)])

        # 4. 聚类逻辑
        # 计算平均行高作为标尺
        avg_h = np.mean([b[3] - b[2] for b in formatted_boxes]) if formatted_boxes else 30
        grouped = []
        used = [False] * len(formatted_boxes)

        for i in range(len(formatted_boxes)):
            if used[i]: continue
            cluster = [formatted_boxes[i]]
            used[i] = True
            found = True

            # 不断吞噬周围的邻居
            while found:
                found = False
                c_x1 = min(b[0] for b in cluster);
                c_x2 = max(b[1] for b in cluster)
                c_y1 = min(b[2] for b in cluster);
                c_y2 = max(b[3] for b in cluster)

                for j in range(len(formatted_boxes)):
                    if used[j]: continue
                    bx1, bx2, by1, by2 = formatted_boxes[j]

                    # 距离判定
                    dx = max(0, c_x1 - bx2, bx1 - c_x2)
                    dy = max(0, c_y1 - by2, by1 - c_y2)

                    # 判定阈值：纵向宽松(1.5倍行高)，横向严格(0.8倍行高)
                    if dy < avg_h * 1.5 and dx < avg_h * 0.8:
                        # 物理墙检测
                        p_start = (int((c_x1 + c_x2) / 2), int((c_y1 + c_y2) / 2))
                        p_end = (int((bx1 + bx2) / 2), int((by1 + by2) / 2))

                        if not self._is_blocked(edges, p_start, p_end):
                            cluster.append(formatted_boxes[j])
                            used[j] = True
                            found = True

            # 保存该簇的整体范围
            grouped.append({
                'box': (min(b[0] for b in cluster), max(b[1] for b in cluster),
                        min(b[2] for b in cluster), max(b[3] for b in cluster)),
                'cluster': cluster  # 仅用于调试绘图
            })

        # --- 5. 命中判定与详细可视化 ---
        vis_img = img.copy()  # 调试画布

        # 绘图层1: 所有原始火柴盒 (绿色细线)
        for b in formatted_boxes:
            cv2.rectangle(vis_img, (b[0], b[2]), (b[1], b[3]), (0, 255, 0), 1)

        final_crop = None
        target_box = None

        for g in grouped:
            gx1, gx2, gy1, gy2 = g['box']

            # 绘图层2: 聚类簇 (青色中线)
            cv2.rectangle(vis_img, (gx1, gy1), (gx2, gy2), (255, 255, 0), 2)

            # 命中检查 (容错 40px)
            if (gx1 - 40) <= cx <= (gx2 + 40) and (gy1 - 40) <= cy <= (gy2 + 40):
                print(f"✅ [Mode 2] Hit cluster with {len(g['cluster'])} boxes")
                target_box = (gx1, gx2, gy1, gy2)

                # 绘图层3: 选中的目标 (红色粗线)
                cv2.rectangle(vis_img, (gx1, gy1), (gx2, gy2), (0, 0, 255), 4)

                # 计算 Padding 并裁切
                pad_w = int((gx2 - gx1) * 0.1) + 15
                pad_h = int((gy2 - gy1) * 0.1) + 15
                h, w = img.shape[:2]
                x1, y1 = max(0, gx1 - pad_w), max(0, gy1 - pad_h)
                x2, y2 = min(w, gx2 + pad_w), min(h, gy2 + pad_h)
                final_crop = img[y1:y2, x1:x2]
                break  # 命中一个即可退出

        # 保存 Mode 2 的丰富调试图
        # 画出点击点
        cv2.circle(vis_img, (cx, cy), 6, (255, 0, 255), -1)
        cv2.putText(vis_img, "Mode 2: Green=Raw, Cyan=Cluster, Red=Selected", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite("debug_mode2_easyocr.png", vis_img)

        return final_crop

    def _is_blocked(self, edges, p1, p2):
        """ 物理墙检测：两点连线上是否有大量边缘点 """
        num_samples = 20
        pts_x = np.linspace(p1[0], p2[0], num_samples).astype(int)
        pts_y = np.linspace(p1[1], p2[1], num_samples).astype(int)
        hits = 0
        h, w = edges.shape[:2]

        for i in range(num_samples):
            px, py = pts_x[i], pts_y[i]
            if 0 <= px < w and 0 <= py < h:
                if edges[py, px] > 0: hits += 1

        # 超过 25% 的路径点踩在边缘上，视为阻隔
        return hits > (num_samples * 0.25)

    def _save_debug_and_return(self, img, x1, y1, x2, y2, ox, oy, suffix=""):
        """ 通用调试保存与裁切函数 """
        h, w = img.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        debug_img = img.copy()
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.circle(debug_img, (int(ox), int(oy)), 7, (255, 0, 0), -1)
        cv2.putText(debug_img, f"Mode: {suffix}", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 统一调试文件名格式
        cv2.imwrite(f"debug_{suffix}.png", debug_img)

        return img[y1:y2, x1:x2]