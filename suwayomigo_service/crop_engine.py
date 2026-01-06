import cv2
import numpy as np


class MangaCropEngine:
    def __init__(self, easyocr_reader=None):
        self.reader = easyocr_reader

    def get_smart_crop(self, image_bytes, click_x_rel, click_y_rel):
        """
        V10.0: 引入形状规则度判定与膨胀碰撞聚类
        """
        # --- 0. 解码与预处理 ---
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return None
        h, w = img.shape[:2]

        cx = int(click_x_rel * w) if 0 < click_x_rel < 1 else int(click_x_rel)
        cy = int(click_y_rel * h) if 0 < click_y_rel < 1 else int(click_y_rel)

        # === Mode 0: 客户端手动圈选模式 ===
        if cx == 0 and cy == 0:
            print("🚀 [Mode 0] Manual/Bypass Mode triggered.")
            # 同样调用调试保存函数，标记为 mode0
            return self._save_debug_and_return(img, 0, 0, w, h, 0, 0, "mode0_manual")

        # --- 1. 自动纠偏 ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        search_radius = 20
        min_y, max_y = max(0, cy - search_radius), min(h, cy + search_radius)
        min_x, max_x = max(0, cx - search_radius), min(w, cx + search_radius)
        sub = gray[min_y:max_y, min_x:max_x]
        blurred_sub = cv2.GaussianBlur(sub, (5, 5), 0)
        if blurred_sub.size > 0:
            _, max_val, _, max_loc = cv2.minMaxLoc(blurred_sub)
            if max_val > 180:
                cx, cy = min_x + max_loc[0], min_y + max_loc[1]

        # --- 2. 气泡探测 (Mode 1) ---
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        flood_filled = gray.copy()
        cv2.floodFill(flood_filled, ff_mask, (cx, cy), 255, (18,), (18,), cv2.FLOODFILL_FIXED_RANGE)
        bubble_mask = ff_mask[1:-1, 1:-1] * 255
        cnts, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        is_valid_bubble = False
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            # 使用更严格的形状分析
            if self._is_bubble_shape(c, w, h):
                is_valid_bubble = True

        # === Mode 1: 几何气泡模式 ===
        if is_valid_bubble:
            print(f"🎯 [Mode 1] Shape Validated Bubble at ({cx}, {cy})")
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
            closed_mask = cv2.morphologyEx(bubble_mask, cv2.MORPH_CLOSE, kernel)
            cnts_closed, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if cnts_closed:
                c = max(cnts_closed, key=cv2.contourArea)
                x, y, rw, rh = cv2.boundingRect(c)
                return self._save_debug_and_return(img, x - 10, y - 10, x + rw + 10, y + rh + 10, cx, cy,
                                                   "mode1_bubble")

        # === Mode 2: EasyOCR 膨胀聚类模式 ===
        print(f"🧠 [Mode 2] Switching to Expansion Clustering at ({cx}, {cy})")
        easy_res = self._try_easyocr_logic(img, gray, cx, cy)
        if easy_res is not None:
            return easy_res

        # === Mode 3: 动态比例保底 ===
        print(f"🩹 [Mode 3] Fallback to Proportional at ({cx}, {cy})")
        fw, fh = int(w * 0.6), int(h * 0.8)
        x1 = max(0, min(w - fw, cx - fw // 2))
        y1 = max(0, min(h - fh, cy - fh // 2))
        return self._save_debug_and_return(img, x1, y1, x1 + fw, y1 + fh, cx, cy, "mode3_fallback")

    def _is_bubble_shape(self, contour, img_w, img_h):
        """
        高级漏气判定：结合凸包实心度和尺寸限制
        """
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # 1. 基础尺寸过滤 (太大或太小都不是气泡)
        if w < 20 or h < 20: return False
        if w > img_w * 0.9 or h > img_h * 0.9: return False  # 几乎占满全屏，通常是背景
        if area > (img_w * img_h * 0.7): return False

        # 2. 凸包实心度 (Convex Solidity)
        # 气泡通常是凸多边形，ConvexHull 与原始轮廓面积差异小
        # 如果是复杂的背景漏气，边缘会很杂乱，Solidity 会很低
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: return False
        solidity = area / hull_area

        # 3. 矩形度 (Extent)
        # 气泡如果是圆角矩形或椭圆，面积不会填满外接矩形
        rect_area = w * h
        extent = area / rect_area

        # 判定逻辑：
        # - Solidity < 0.7: 形状极度不规则（像蜘蛛网），判定为漏气
        # - Extent > 0.95: 极度方正且占地大，可能是漫画格子的边框而非气泡
        if solidity < 0.75:
            print(f"🛡️ [Leak Check] Low solidity ({solidity:.2f}). Rejected.")
            return False

        # 可选：椭圆拟合检查 (对于非常标准的椭圆气泡)
        if len(contour) >= 5:
            (ex, ey), (ew, eh), angle = cv2.fitEllipse(contour)
            ellipse_area = (np.pi * ew * eh) / 4
            # 如果拟合椭圆面积和实际面积极其接近，那必定是气泡
            if 0.85 < (area / ellipse_area) < 1.15:
                return True

        return True

    def _try_easyocr_logic(self, img, gray, cx, cy):
        """
        Mode 2 优化版：膨胀交叉聚类 (Expand & Intersect)
        """
        if not self.reader: return None
        edges = cv2.Canny(gray, 70, 200)

        # 1. OCR 探测
        try:
            horizontal_list, _ = self.reader.detect(img, text_threshold=0.3)
            raw_boxes = horizontal_list[0] if horizontal_list else []
        except:
            return None
        if not raw_boxes: return None

        # 2. 格式化并生成膨胀框
        # box_data: [{'orig': [x1,y1,x2,y2], 'expand': [x1,y1,x2,y2]}, ...]
        box_data = []
        for b in raw_boxes:
            # 兼容不同格式
            bx1, bx2, by1, by2 = 0, 0, 0, 0
            if len(b) == 4:
                bx1, bx2, by1, by2 = b
            else:
                xs, ys = [p[0] for p in b], [p[1] for p in b]
                bx1, bx2, by1, by2 = min(xs), max(xs), min(ys), max(ys)

            # 计算膨胀量 (1.1倍尺寸 -> 单边增加 5% 宽，7.5% 高，针对竖排文字)
            w_box, h_box = bx2 - bx1, by2 - by1
            pad_x = int(w_box * 0.05) + 2  # +2px 保底
            pad_y = int(h_box * 0.10) + 2  # 竖排稍微多扩一点

            expand_box = [bx1 - pad_x, bx2 + pad_x, by1 - pad_y, by2 + pad_y]
            box_data.append({'orig': [bx1, bx2, by1, by2], 'expand': expand_box})

        # 3. 交叉聚类 (Iterative Merge)
        # 只要膨胀框相交 且 无物理阻断，就合并
        # 我们使用一个简单的并查集思路，或者反复遍历直到稳定

        clusters = [[i] for i in range(len(box_data))]  # 初始：每个box一个簇

        changed = True
        while changed:
            changed = False
            new_clusters = []
            while clusters:
                current_cluster = clusters.pop(0)
                merged_occurred = False

                # 尝试将 current_cluster 与剩余的 clusters 合并
                # 我们逆向遍历以便安全删除
                for i in range(len(clusters) - 1, -1, -1):
                    target_cluster = clusters[i]

                    # 判断两个簇是否应该合并
                    if self._should_merge_clusters(current_cluster, target_cluster, box_data, edges):
                        current_cluster.extend(target_cluster)
                        clusters.pop(i)
                        merged_occurred = True
                        changed = True

                new_clusters.append(current_cluster)
            clusters = new_clusters

        # 4. 生成最终包围盒并判定命中
        vis_img = img.copy()
        final_res = None

        for cluster_indices in clusters:
            # 计算该簇的原始包围盒 (使用 orig 坐标)
            all_origs = [box_data[i]['orig'] for i in cluster_indices]
            gx1 = min(b[0] for b in all_origs)
            gx2 = max(b[1] for b in all_origs)
            gy1 = min(b[2] for b in all_origs)
            gy2 = max(b[3] for b in all_origs)

            # 绘图：Raw(绿), Cluster(青)
            cv2.rectangle(vis_img, (gx1, gy1), (gx2, gy2), (255, 255, 0), 2)
            for idx in cluster_indices:
                b = box_data[idx]['orig']
                cv2.rectangle(vis_img, (b[0], b[2]), (b[1], b[3]), (0, 255, 0), 1)

            # 命中判定 (40px 容错)
            if (gx1 - 40) <= cx <= (gx2 + 40) and (gy1 - 40) <= cy <= (gy2 + 40):
                print(f"✅ [Mode 2] Hit cluster with {len(cluster_indices)} boxes")
                cv2.rectangle(vis_img, (gx1, gy1), (gx2, gy2), (0, 0, 255), 4)

                pad_w = int((gx2 - gx1) * 0.1) + 15
                pad_h = int((gy2 - gy1) * 0.1) + 15
                x1, y1 = max(0, gx1 - pad_w), max(0, gy1 - pad_h)
                x2, y2 = min(img.shape[1], gx2 + pad_w), min(img.shape[0], gy2 + pad_h)
                final_res = img[y1:y2, x1:x2]
                # 这里不break，为了画完所有的调试框，但会保留命中的结果

        cv2.circle(vis_img, (cx, cy), 6, (255, 0, 255), -1)
        cv2.imwrite("debug_mode2_easyocr.png", vis_img)
        return final_res

    def _should_merge_clusters(self, cluster_a, cluster_b, box_data, edges):
        """
        判断两个簇是否物理接触。
        逻辑：只要簇A中有任意一个box 与 簇B中任意一个box 的【膨胀框】相交，且无墙，即合并。
        """
        for ia in cluster_a:
            box_a = box_data[ia]
            for ib in cluster_b:
                box_b = box_data[ib]

                # 1. 检查膨胀框是否相交
                ax1, ax2, ay1, ay2 = box_a['expand']
                bx1, bx2, by1, by2 = box_b['expand']

                # 矩形重叠判定
                overlap = not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

                if overlap:
                    # 2. 物理墙判定 (使用原始中心连线)
                    orig_a = box_a['orig']
                    orig_b = box_b['orig']
                    pa = (int((orig_a[0] + orig_a[1]) / 2), int((orig_a[2] + orig_a[3]) / 2))
                    pb = (int((orig_b[0] + orig_b[1]) / 2), int((orig_b[2] + orig_b[3]) / 2))

                    if not self._is_blocked(edges, pa, pb):
                        return True
        return False

    def _is_blocked(self, edges, p1, p2):
        num = 20
        pts_x = np.linspace(p1[0], p2[0], num).astype(int)
        pts_y = np.linspace(p1[1], p2[1], num).astype(int)
        hits = 0
        h, w = edges.shape[:2]
        for i in range(num):
            px, py = pts_x[i], pts_y[i]
            if 0 <= px < w and 0 <= py < h:
                if edges[py, px] > 0: hits += 1
        return hits > (num * 0.25)

    def _save_debug_and_return(self, img, x1, y1, x2, y2, ox, oy, suffix=""):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        debug_img = img.copy()
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.circle(debug_img, (int(ox), int(oy)), 7, (255, 0, 0), -1)
        cv2.putText(debug_img, suffix, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(f"debug_{suffix}.png", debug_img)
        return img[y1:y2, x1:x2]