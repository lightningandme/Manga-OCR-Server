import base64
import os
import sys
import cv2
import numpy as np
import uvicorn
from PIL import Image
from fastapi import FastAPI, Body
from dotenv import load_dotenv

# 关键：将项目根目录加入系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from manga_ocr import MangaOcr
from janome.tokenizer import Tokenizer

from openai import OpenAI
import time

# 加载当前目录下的 .env 文件
load_dotenv()

# 从环境变量中读取
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")
# --- 配置 DeepSeek ---
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

app = FastAPI()

# 初始化 OCR 模型
print("Loading Manga-OCR model...")
mocr = MangaOcr()

# 初始化 Janome 分词器 (本地运行，极快)
print("Initializing Tokenizer...")
tokenizer = Tokenizer()


def get_smart_crop(image_bytes, click_x_rel, click_y_rel):
    """
        针对前端固定 600x600 输入优化的切图算法
        click_x_rel, click_y_rel: 点击点在 600x600 局部图中的相对坐标
        """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None
    h, w = img.shape[:2]  # 此时 h, w 理论上都是 600

    cx, cy = int(click_x_rel), int(click_y_rel)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 1. 自动纠偏（固定半径 20px） ---
    search_radius = 20
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    min_y, max_y = max(0, cy - search_radius), min(h, cy + search_radius)
    min_x, max_x = max(0, cx - search_radius), min(w, cx + search_radius)
    sub = blurred[min_y:max_y, min_x:max_x]

    if sub.size > 0:
        _, max_val, _, max_loc = cv2.minMaxLoc(sub)
        if max_val > 180:
            cx, cy = min_x + max_loc[0], min_y + max_loc[1]

    # --- 2. 魔法棒探测 ---
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    flood_filled = gray.copy()
    # 稍微放宽容差值（18, 18），更适合 600px 的文字密度
    cv2.floodFill(flood_filled, ff_mask, (cx, cy), 255, (18,), (18,), cv2.FLOODFILL_FIXED_RANGE)
    bubble_mask = ff_mask[1:-1, 1:-1] * 255

    cnts, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    is_leaking = False
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, rw, rh = cv2.boundingRect(c)
        # 允许气泡占据宽度的 85% 或高度的 85%，面积不超过总面积的 40%
        if rw > w * 0.85 or rh > h * 0.95 or cv2.contourArea(c) > (w * h * 0.7):
            is_leaking = True
    else:
        is_leaking = True

    # --- 3. 逻辑分支 ---
    if not is_leaking:
        # 情况 A：气泡捕获，使用固定 25px 闭运算
        print("--- 模式：对话气泡捕获 ---")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        bubble_mask = cv2.morphologyEx(bubble_mask, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        x, y, rw, rh = cv2.boundingRect(c)
        pad = 10
    else:
        # --- 模式：定向流向聚合 (针对多列排版优化) ---
        print("--- 模式：定向流向聚合 ---")

        # 1. 依然保留高光+边缘的双通道提取（这部分效果很好）
        _, bright_mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(gray, 60, 180)
        combined_features = cv2.bitwise_or(edges, bright_mask)

        # 2. 稍微减小一点膨胀力度，让字与字之间先保持一点距离
        # 之前是 (40, 5) 和 (5, 40)，这里稍微收敛一点，依赖后面的逻辑去连
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 30))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
        dilated = cv2.dilate(combined_features, kernel_v)
        dilated = cv2.dilate(dilated, kernel_h)

        # 保存调试图
        cv2.imwrite("debug_radar_mask.png", dilated)

        r_cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 收集所有候选块，并过滤掉太大的（比如漫画边框）
        candidates = []
        seed_idx = -1

        for i, rc in enumerate(r_cnts):
            rx, ry, rrw, rrh = cv2.boundingRect(rc)
            # 过滤掉几乎占满全图的边框噪音
            if rrw > w * 0.95 or rrh > h * 0.95:
                continue

            candidates.append((rx, ry, rrw, rrh))

            # 找到点击点所在的块
            if rx <= cx <= rx + rrw and ry <= cy <= ry + rrh:
                seed_idx = len(candidates) - 1

        if seed_idx != -1:
            # --- 【核心改进】：动态生长 + 对齐优先聚合 ---

            # 初始化聚合集合
            merged_indices = {seed_idx}

            # 两个核心阈值
            # 1. FLOW_GAP: 顺着文字流向（如竖排的上下）允许的最大间断距离
            FLOW_GAP = 120  # 稍微给大点，应对一些艺术排版的大空隙
            # 2. CROSS_GAP: 垂直于流向（如竖排的换列）允许的最大偏离距离
            CROSS_GAP = 15  # 必须很小，防止连到隔壁列去

            has_new_merge = True

            while has_new_merge:
                has_new_merge = False

                # 1. 获取当前大团块的尺寸和边界
                current_rects = [candidates[i] for i in merged_indices]
                min_x = min([r[0] for r in current_rects])
                min_y = min([r[1] for r in current_rects])
                max_x = max([r[0] + r[2] for r in current_rects])
                max_y = max([r[1] + r[3] for r in current_rects])
                curr_w = max_x - min_x
                curr_h = max_y - min_y

                # 2. 动态判断当前大团块的流向
                # 随着合并的进行，is_vertical 可能会从 False 变成 True
                is_vertical = curr_h > curr_w * 1.1
                is_horizontal = curr_w > curr_h * 1.1
                # 如果都不是，说明还是个方块（ambiguous），此时依靠对齐度来判断

                # 3. 遍历寻找可以吞噬的邻居
                for i in range(len(candidates)):
                    if i in merged_indices: continue

                    ox, oy, ow, oh = candidates[i]
                    ox2, oy2 = ox + ow, oy + oh

                    should_merge = False

                    # 计算投影重叠度（判断对齐）
                    # X轴重叠长度 / 较小的那一个宽度
                    overlap_x = max(0, min(max_x, ox2) - max(min_x, ox))
                    ratio_align_v = overlap_x / min(curr_w, ow) if min(curr_w, ow) > 0 else 0

                    # Y轴重叠长度 / 较小的那一个高度
                    overlap_y = max(0, min(max_y, oy2) - max(min_y, oy))
                    ratio_align_h = overlap_y / min(curr_h, oh) if min(curr_h, oh) > 0 else 0

                    # 计算间距
                    dist_x = max(0, max(min_x, ox) - min(max_x, ox2))
                    dist_y = max(0, max(min_y, oy) - min(max_y, oy2))

                    # === 判定逻辑 A: 明确的竖排模式 ===
                    if is_vertical:
                        # 必须在X轴高度对齐 (同一列) 且 Y轴距离在允许范围内
                        # 或者 距离极近的标点符号
                        if (ratio_align_v > 0.5 and dist_y < FLOW_GAP) or (dist_x < CROSS_GAP and dist_y < CROSS_GAP):
                            should_merge = True

                    # === 判定逻辑 B: 明确的横排模式 ===
                    elif is_horizontal:
                        # 必须在Y轴高度对齐 (同一行) 且 X轴距离在允许范围内
                        if (ratio_align_h > 0.5 and dist_x < FLOW_GAP) or (dist_x < CROSS_GAP and dist_y < CROSS_GAP):
                            should_merge = True

                    # === 判定逻辑 C: 方块/不定状态 (关键修复点) ===
                    else:
                        # 既不横也不竖，说明是初始种子。
                        # 策略：谁跟我对其最准，我就跟谁连！

                        # 如果跟下方/上方方块 X轴对齐度极高 -> 尝试竖向合并
                        if ratio_align_v > 0.6 and dist_y < FLOW_GAP:
                            should_merge = True

                        # 如果跟左边/右边方块 Y轴对齐度极高 -> 尝试横向合并
                        elif ratio_align_h > 0.6 and dist_x < FLOW_GAP:
                            should_merge = True

                        # 保底：如果距离特别近，不管对齐不对齐都吸进来 (处理标点)
                        elif dist_x < 20 and dist_y < 20:
                            should_merge = True

                    if should_merge:
                        merged_indices.add(i)
                        has_new_merge = True
                        # 只要有一个新块加进来，就会改变大团块的宽高比
                        # 下一次循环就会根据新的形状重新判断 is_vertical/is_horizontal
                        # 从而触发“连锁反应”
                        break  # 重新计算大包围盒，进入下一次 while 循环

            # 最终输出
            x, y = min_x, min_y
            rw, rh = max_x - min_x, max_y - min_y
            pad = 20
        else:
            # 保底
            print("--- 模式：触发保底识别范围 ---")
            # --- 改进的动态保底逻辑 ---
            # 定义保底框占原图的比例（例如：宽占 50%，高占 25%）
            fallback_w = int(w * 0.6)
            fallback_h = int(h * 0.8)

            # 计算起始坐标，使点击点处于框的中心
            x = cx - (fallback_w // 2)
            y = cy - (fallback_h // 2)
            rw, rh, pad = fallback_w, fallback_h, 0

    # --- 4. 最终裁剪并确保不越界 ---
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + rw + pad), min(h, y + rh + pad)

    # --- 调试代码开始 (插入在 return 之前) ---
    debug_img = img.copy()
    # 画出最终确定的矩形框 (红色，粗细为 3)
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    # 画出点击的原始坐标点 (蓝色小圆点)
    cv2.circle(debug_img, (int(click_x_rel), int(click_y_rel)), 5, (255, 0, 0), -1)

    # 保存调试图片到后端服务器目录下
    cv2.imwrite("debug_result.png", debug_img)

    return img[y1:y2, x1:x2]

def analyze_text(text: str):
    """
    对日语文本进行分词，并提取学习相关的元数据
    """
    results = []
    # 运行分词
    tokens = tokenizer.tokenize(text)

    for token in tokens:
        # pos 格式通常为: 名詞,一般,*,*
        pos_details = token.part_of_speech.split(',')
        main_pos = pos_details[0]  # 主词性

        # 过滤掉标点符号和空白，只保留有意义的词汇
        if main_pos in ['記号', '助詞', '助動詞'] and token.surface in [' ', '　', '。', '、','．','！','？','：']:
            continue

        results.append({
            "s": token.surface,  # 表面形：漫画里显示的原文
            "b": token.base_form,  # 原型：查词典用的原始形态
            "p": main_pos,  # 词性：名词、动词、形容词等
            "r": token.reading  # 读音：片假名读音 (可选)
        })
    return results


def get_ai_translation(text: str, manga_name: str):
    if not text.strip():
        return ""

    try:
        start_time = time.time()
        # 使用极简 Prompt：不要求解释，只要求地道翻译和核心词原型
        # noinspection PyTypeChecker

        # 按照你提供的 Prompt 模板构建 System Content
        system_content = (
            f"你是一位精通多门语言的漫画翻译专家。"
            f"当前语境：正在阅读漫画《{manga_name}》。"
            f"请根据该作品的风格（热血/少女/日常等）、背景设定和角色身份，"
            f"将输入的日文翻译成地道、流畅的中文。直接返回译文。"
        )

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": text},
            ],
            stream=False,
            temperature=0.3,  # 降低随机性，让翻译更稳定
            max_tokens=150  # 限制输出长度，减少传输耗时
        )
        duration = time.time() - start_time
        print(f"AI 响应耗时: {duration:.2f}s (作品: {manga_name})")
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"翻译出错了: {str(e)}"


# 缓存最近一次的 OCR 文本和漫画名
last_ocr_text = ""
last_manga_name = "General"
@app.post("/ocr")
async def perform_ocr(payload: dict = Body(...)):
    global last_ocr_text, last_manga_name  # <--- 修改这里，加入 last_manga_name
    last_ocr_text = ""  # 每次识别新图前先清空旧缓存
    img_b64 = payload.get("image")
    # 获取 Android 传来的点击坐标
    click_x = payload.get("x", 0)
    click_y = payload.get("y", 0)
    manga_name = payload.get("mangaName", "General")

    last_manga_name = manga_name  # <--- 核心修改：将本次漫画名存入缓存
    if not img_b64:
        return {"status": "error", "message": "No image data"}

    try:
        # 1. 解码与识别
        img_data = base64.b64decode(img_b64)

        # --- 智能切图核心调用 ---
        # 注意：这里的 img_data 是 Android 传来的 400x400 或 600x600 的局部图
        # 这里的 click_x/y 应该是相对于这张局部图的坐标
        smart_img_mat = get_smart_crop(img_data, click_x, click_y)

        # 将 OpenCV 的 Mat 转回 PIL Image 给 Manga-OCR 使用
        smart_img_rgb = cv2.cvtColor(smart_img_mat, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(smart_img_rgb)

        # 后续 OCR 逻辑不变
        text = mocr(image)
        last_ocr_text = text  # 存入缓存

        words = analyze_text(text)

        # 核心：这里不再调用 AI 翻译，直接返回，速度提升 200%
        return {
            "status": "success",
            "text": text,
            "words": words,
            "translation": ""  # 初始为空
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/get_translation")
async def get_translation():
    global last_ocr_text, last_manga_name  # <--- 声明读取这两个全局变量
    if not last_ocr_text:
        return {"translation": "未检测到待翻译文字"}

    # 调用时传入缓存的漫画名
    translation = get_ai_translation(last_ocr_text, last_manga_name)
    return {"translation": translation}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=12233)