import cv2
import numpy as np
import time


class MangaCropEngine:
    def __init__(self, easyocr_reader):
        """
        :param easyocr_reader: 外部传入的 easyocr.Reader 实例
        """
        self.reader = easyocr_reader

    def get_smart_crop(self, img_cv2, click_x, click_y):
        if img_cv2 is None:
            return None

        h_img, w_img = img_cv2.shape[:2]

        # --- 【模式 0】客户端精准圈选模式 ---
        if click_x == 0 and click_y == 0:
            print("🚀 [Mode 0] Client Manual Mode. Returning full image.")
            return img_cv2

        # --- 【模式 1】气泡框识别模式 (快速 OpenCV 方案) ---
        bubble_crop = self._try_bubble_mode(img_cv2, click_x, click_y)
        if bubble_crop is not None:
            print("🎯 [Mode 1] Bubble Contour Success.")
            return bubble_crop

        # --- 【模式 2】Smart OCR 聚合模式 (EasyOCR + 物理墙) ---
        print("🧠 [Mode 1 Failed] Switching to Mode 2: EasyOCR Aggregation...")
        semantic_crop = self._try_easyocr_aggregation(img_cv2, click_x, click_y)
        if semantic_crop is not None:
            print("✅ [Mode 2] EasyOCR Aggregation Success.")
            return semantic_crop

        # --- 【模式 3】动态比例裁切保底 ---
        print("🩹 [Mode 2 Failed] Switching to Mode 3: Dynamic Fallback.")
        return self._fallback_dynamic_crop(img_cv2, click_x, click_y)

    # ================= 策略实现内部函数 =================

    def _try_bubble_mode(self, img, x, y):
        """ 模式 1：基于 FloodFill 的气泡查找 """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 建立 Mask
        mask = np.zeros((h + 2, w + 2), np.uint8)
        # 尝试 FloodFill，设定较低的阈值以适应漫画纸张背景
        # loDiff/upDiff=20 是为了适应轻微的背景网点
        temp_img = img.copy()
        cv2.floodFill(temp_img, mask, (int(x), int(y)), (255, 255, 255), (20, 20, 20), (20, 20, 20),
                      8 | cv2.FLOODFILL_MASK_ONLY)

        # 提取填充区域
        actual_mask = mask[1:-1, 1:-1]
        contours, _ = cv2.findContours(actual_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(cnt)

            # 校验：气泡不能太小（排除杂点），也不能太大（防止充满全图）
            if bw > 30 and bh > 30 and bw < w * 0.9 and bh < h * 0.9:
                # 增加 5% 的 Padding
                p_w, p_h = int(bw * 0.05), int(bh * 0.05)
                x1, y1 = max(0, bx - p_w), max(0, by - p_h)
                x2, y2 = min(w, bx + bw + p_w), min(h, by + bh + p_h)

                # 调试图输出
                debug_vis = img.copy()
                cv2.rectangle(debug_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite("debug_mode1_bubble.png", debug_vis)

                return img[y1:y2, x1:x2]
        return None

    def _try_easyocr_aggregation(self, img, x, y):
        """ 模式 2：EasyOCR 语义聚合 + 物理墙探测 """
        h_img, w_img = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)  # 预计算物理墙

        # 1. 检测原始火柴盒
        horizontal_list, _ = self.reader.detect(img, text_threshold=0.5)
        raw_boxes = horizontal_list[0] if horizontal_list else []
        if not raw_boxes: return None

        # 2. 聚类
        avg_h = np.mean([b[3] - b[2] for b in raw_boxes]) if raw_boxes else 30
        grouped = []
        used = [False] * len(raw_boxes)

        for i in range(len(raw_boxes)):
            if used[i]: continue
            cluster = [raw_boxes[i]]
            used[i] = True

            found = True
            while found:
                found = False
                c_x1, c_x2 = min(b[0] for b in cluster), max(b[1] for b in cluster)
                c_y1, c_y2 = min(b[2] for b in cluster), max(b[3] for b in cluster)

                for j in range(len(raw_boxes)):
                    if used[j]: continue
                    bx1, bx2, by1, by2 = raw_boxes[j]
                    dx = max(0, c_x1 - bx2, bx1 - c_x2)
                    dy = max(0, c_y1 - by2, by1 - c_y2)

                    # 基础距离判定 + 物理墙探测
                    if dy < avg_h * 1.2 and dx < avg_h * 0.4:
                        p1 = (int((c_x1 + c_x2) / 2), int((c_y1 + c_y2) / 2))
                        p2 = (int((bx1 + bx2) / 2), int((by1 + by2) / 2))
                        if not self._is_blocked(edges, p1, p2):
                            cluster.append(raw_boxes[j])
                            used[j] = True
                            found = True

            grouped.append((min(b[0] for b in cluster), max(b[1] for b in cluster),
                            min(b[2] for b in cluster), max(b[3] for b in cluster)))

        # 3. 匹配点击点
        best_box = None
        for box in grouped:
            x1, x2, y1, y2 = map(int, box)
            if (x1 - 15) <= x <= (x2 + 15) and (y1 - 15) <= y <= (y2 + 15):
                best_box = (x1, x2, y1, y2)
                break

        if best_box:
            x1, x2, y1, y2 = best_box
            pw, ph = int((x2 - x1) * 0.1) + 10, int((y2 - y1) * 0.05) + 10
            # 调试图
            debug_vis = img.copy()
            cv2.rectangle(debug_vis, (max(0, x1 - pw), max(0, y1 - ph)), (min(w_img, x2 + pw), min(h_img, y2 + ph)),
                          (255, 0, 0), 2)
            cv2.imwrite("debug_mode2_easyocr.png", debug_vis)
            return img[max(0, y1 - ph):min(h_img, y2 + ph), max(0, x1 - pw):min(w_img, x2 + pw)]

        return None

    def _is_blocked(self, edges, p1, p2):
        """ 物理墙探测辅助函数 """
        num_samples = 15
        pts_x = np.linspace(p1[0], p2[0], num_samples).astype(int)
        pts_y = np.linspace(p1[1], p2[1], num_samples).astype(int)
        hits = 0
        h, w = edges.shape
        for i in range(num_samples):
            px, py = pts_x[i], pts_y[i]
            if 0 <= px < w and 0 <= py < h:
                if edges[py, px] > 0: hits += 1
                if hits >= 2: return True
        return False

    def _fallback_dynamic_crop(self, img, x, y):
        """ 模式 3：0.4W x 0.6H 动态比例裁剪 """
        h, w = img.shape[:2]
        cw, ch = int(w * 0.4), int(h * 0.6)

        x1 = max(0, int(x - cw / 2))
        y1 = max(0, int(y - ch / 2))
        x2 = min(w, x1 + cw)
        y2 = min(h, y1 + ch)

        # 边界补全
        if x2 == w: x1 = max(0, x2 - cw)
        if y2 == h: y1 = max(0, y2 - ch)

        debug_vis = img.copy()
        cv2.rectangle(debug_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite("debug_mode3_fallback.png", debug_vis)

        return img[y1:y2, x1:x2]