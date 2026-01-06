import cv2
import numpy as np
import os


class MangaCropEngine:
    def __init__(self, easyocr_reader=None):
        self.reader = easyocr_reader

    def get_smart_crop(self, image_bytes, click_x_rel, click_y_rel):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return None
        h, w = img.shape[:2]

        # 1. 坐标转换与【自动纠偏】
        cx, cy = int(click_x_rel), int(click_y_rel)
        if cx == 0 and cy == 0: return img

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        search_radius = 20
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        min_y, max_y = max(0, cy - search_radius), min(h, cy + search_radius)
        min_x, max_x = max(0, cx - search_radius), min(w, cx + search_radius)
        sub = blurred[min_y:max_y, min_x:max_x]
        if sub.size > 0:
            _, max_val, _, max_loc = cv2.minMaxLoc(sub)
            if max_val > 180:
                cx, cy = min_x + max_loc[0], min_y + max_loc[1]

        # --- 2. 魔法棒探测与漏气判定 ---
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        flood_filled = gray.copy()
        cv2.floodFill(flood_filled, ff_mask, (cx, cy), 255, (18,), (18,), cv2.FLOODFILL_FIXED_RANGE)
        bubble_mask = ff_mask[1:-1, 1:-1] * 255
        cnts, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        is_leaking = True
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            solidity = area / float(bw * bh) if (bw * bh) > 0 else 0
            if not (bw > w * 0.8 or bh > h * 0.8 or area > (w * h * 0.6) or (solidity > 0.9 and area > (w * h * 0.3))):
                is_leaking = False

        # --- 3. 逻辑分流 (修复返回链条) ---
        if not is_leaking:
            print("--- Mode: Bubble Capture ---")
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
            bubble_mask = cv2.morphologyEx(bubble_mask, cv2.MORPH_CLOSE, kernel)
            cnts, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                x, y, rw, rh = cv2.boundingRect(c)
                # 直接通过调试函数返回，确保不再向下执行
                return self._save_debug_and_return(img, x - 10, y - 10, x + rw + 10, y + rh + 10, cx, cy, "mode1")

        # 如果漏气或 Mode 1 失败，进入聚合模式
        print("--- Mode: Semantic/Flow Aggregation ---")
        easy_res = self._try_easyocr_logic(img, gray, cx, cy)
        if easy_res is not None:
            return easy_res

        # 最后的兜底：定向流向聚合 或 动态比例保底
        return self._original_flow_aggregation(img, gray, cx, cy)

    def _try_easyocr_logic(self, img, gray, cx, cy):
        if not self.reader: return None
        h_img, w_img = img.shape[:2]
        edges = cv2.Canny(gray, 70, 200)  # 调高阈值，减少背景噪音干扰

        horizontal_list, _ = self.reader.detect(img, text_threshold=0.3)
        raw_boxes = horizontal_list[0] if horizontal_list else []
        if not raw_boxes: return None

        # 格式标准化：确保是 [x1, x2, y1, y2]
        formatted_boxes = []
        for b in raw_boxes:
            if len(b) == 4:  # [x1, x2, y1, y2]
                formatted_boxes.append(b)
            elif len(b) == 2 and len(b[0]) == 2:  # [[x,y],[x,y],[x,y],[x,y]]
                xs = [p[0] for p in b]
                ys = [p[1] for p in b]
                formatted_boxes.append([min(xs), max(xs), min(ys), max(ys)])

        avg_h = np.mean([b[3] - b[2] for b in formatted_boxes]) if formatted_boxes else 30
        grouped = []
        used = [False] * len(formatted_boxes)

        for i in range(len(formatted_boxes)):
            if used[i]: continue
            cluster = [formatted_boxes[i]]
            used[i] = True
            found = True
            while found:
                found = False
                c_x1, c_x2 = min(b[0] for b in cluster), max(b[1] for b in cluster)
                c_y1, c_y2 = min(b[2] for b in cluster), max(b[3] for b in cluster)
                for j in range(len(formatted_boxes)):
                    if used[j]: continue
                    bx1, bx2, by1, by2 = formatted_boxes[j]
                    dx = max(0, c_x1 - bx2, bx1 - c_x2)
                    dy = max(0, c_y1 - by2, by1 - c_y2)

                    if dy < avg_h * 1.5 and dx < avg_h * 0.8:  # 放宽聚合距离
                        p_s = (int((c_x1 + c_x2) / 2), int((c_y1 + c_y2) / 2))
                        p_e = (int((bx1 + bx2) / 2), int((by1 + by2) / 2))
                        if not self._is_blocked(edges, p_s, p_e):
                            cluster.append(formatted_boxes[j])
                            used[j] = True
                            found = True

            grouped.append({'box': (min(b[0] for b in cluster), max(b[1] for b in cluster),
                                    min(b[2] for b in cluster), max(b[3] for b in cluster))})

        for g in grouped:
            x1, x2, y1, y2 = g['box']
            # 将容错范围扩大到 40px，解决纠偏后的点击偏移
            if (x1 - 40) <= cx <= (x2 + 40) and (y1 - 40) <= cy <= (y2 + 40):
                pad_w, pad_h = int((x2 - x1) * 0.1) + 15, int((y2 - y1) * 0.1) + 15
                return self._save_debug_and_return(img, x1 - pad_w, y1 - pad_h, x2 + pad_w, y2 + pad_h, cx, cy, "mode2")
        return None

    def _is_blocked(self, edges, p1, p2):
        num_samples = 20
        pts_x = np.linspace(p1[0], p2[0], num_samples).astype(int)
        pts_y = np.linspace(p1[1], p2[1], num_samples).astype(int)
        hits = 0
        h_max, w_max = edges.shape[:2]
        for i in range(num_samples):
            px, py = pts_x[i], pts_y[i]
            if 0 <= px < w_max and 0 <= py < h_max:
                if edges[py, px] > 0: hits += 1
        # 提高碰撞门槛：超过 25% 的采样点撞墙才认为是阻塞，防止网点干扰
        return hits > (num_samples * 0.25)

    def _save_debug_and_return(self, img, x1, y1, x2, y2, ox, oy, suffix=""):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))

        # 只有在返回有效区域时才保存调试图
        debug_img = img.copy()
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.circle(debug_img, (int(ox), int(oy)), 7, (255, 0, 0), -1)
        cv2.putText(debug_img, f"Target: {suffix}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(f"debug_result_{suffix}.png", debug_img)

        return img[y1:y2, x1:x2]

    def _original_flow_aggregation(self, img, gray, cx, cy):
        """
        搬运并优化：定向流向聚合逻辑
        该逻辑通过动态计算团块的长宽比来决定生长方向（竖排/横排/方块）
        """
        h, w = img.shape[:2]

        # 1. 多特征融合：边缘 + 高光 + 局部对比度
        edges = cv2.Canny(gray, 60, 180)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        adaptive_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 15, 8)

        combined_features = cv2.bitwise_or(cv2.bitwise_or(edges, bright_mask), adaptive_mask)

        # 2. 定向形态学膨胀 (3, 30) 和 (30, 3)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 30))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
        dilated = cv2.dilate(combined_features, kernel_v)
        dilated = cv2.dilate(dilated, kernel_h)

        # 保存中间调试图
        cv2.imwrite("debug_radar_mask.png", dilated)

        r_cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        seed_idx = -1
        for i, rc in enumerate(r_cnts):
            rx, ry, rrw, rrh = cv2.boundingRect(rc)
            if rrw > w * 0.95 or rrh > h * 0.95: continue  # 过滤边框
            candidates.append((rx, ry, rrw, rrh))
            if rx <= cx <= rx + rrw and ry <= cy <= ry + rrh:
                seed_idx = len(candidates) - 1

        if seed_idx != -1:
            # --- 【核心逻辑】：动态生长聚合循环 ---
            merged_indices = {seed_idx}
            FLOW_GAP = 120  # 顺着文字流向的最大空隙
            CROSS_GAP = 15  # 垂直流向的严苛限制
            has_new_merge = True

            while has_new_merge:
                has_new_merge = False

                # 计算当前团块的边界
                current_rects = [candidates[i] for i in merged_indices]
                min_x = min([r[0] for r in current_rects])
                min_y = min([r[1] for r in current_rects])
                max_x = max([r[0] + r[2] for r in current_rects])
                max_y = max([r[1] + r[3] for r in current_rects])
                curr_w = max_x - min_x
                curr_h = max_y - min_y

                # 动态判断流向
                is_vertical = curr_h > curr_w * 1.1
                is_horizontal = curr_w > curr_h * 1.1

                for i in range(len(candidates)):
                    if i in merged_indices: continue
                    ox, oy, ow, oh = candidates[i]
                    ox2, oy2 = ox + ow, oy + oh

                    should_merge = False
                    # 计算投影对齐度
                    overlap_x = max(0, min(max_x, ox2) - max(min_x, ox))
                    ratio_align_v = overlap_x / min(curr_w, ow) if min(curr_w, ow) > 0 else 0
                    overlap_y = max(0, min(max_y, oy2) - max(min_y, oy))
                    ratio_align_h = overlap_y / min(curr_h, oh) if min(curr_h, oh) > 0 else 0

                    dist_x = max(0, max(min_x, ox) - min(max_x, ox2))
                    dist_y = max(0, max(min_y, oy) - min(max_y, oy2))

                    # 判定逻辑：竖排/横排/初始态
                    if is_vertical:
                        if (ratio_align_v > 0.5 and dist_y < FLOW_GAP) or (dist_x < CROSS_GAP and dist_y < CROSS_GAP):
                            should_merge = True
                    elif is_horizontal:
                        if (ratio_align_h > 0.5 and dist_x < FLOW_GAP) or (dist_x < CROSS_GAP and dist_y < CROSS_GAP):
                            should_merge = True
                    else:
                        # Ambiguous 状态：优先吸纳对齐度高的邻居
                        if ratio_align_v > 0.6 and dist_y < FLOW_GAP:
                            should_merge = True
                        elif ratio_align_h > 0.6 and dist_x < FLOW_GAP:
                            should_merge = True
                        elif dist_x < 20 and dist_y < 20:
                            should_merge = True

                    if should_merge:
                        merged_indices.add(i)
                        has_new_merge = True
                        break  # 更新边界后重新开始遍历

            # 最终返回合并后的区域，额外给 20px padding
            return self._save_debug_and_return(img, min_x - 20, min_y - 20, max_x + 20, max_y + 20, cx, cy)

        # --- 策略 3: 终极动态比例保底 ---
        print("🩹 [Mode 3] Proportional Fallback.")
        fw, fh = int(w * 0.6), int(h * 0.8)
        # 确保中心点对齐且不越界
        x1 = max(0, min(w - fw, cx - fw // 2))
        y1 = max(0, min(h - fh, cy - fh // 2))
        return self._save_debug_and_return(img, x1, y1, x1 + fw, y1 + fh, cx, cy)
