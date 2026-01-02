import base64
import os
import sys
import cv2
import numpy as np
import uvicorn
from PIL import Image
from fastapi import FastAPI, Body

# 关键：将项目根目录加入系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from manga_ocr import MangaOcr
from janome.tokenizer import Tokenizer

from openai import OpenAI
import time

# --- 配置 DeepSeek ---
client = OpenAI(
    api_key="sk-77aef9337e7648f58754965460002864",
    base_url="https://api.deepseek.com"
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
        # 600px 下，如果选区宽度超过 520 或面积过大，判定为漏水
        if rw > 520 or rh > 520 or cv2.contourArea(c) > (600 * 600 * 0.4):
            is_leaking = True
    else:
        is_leaking = True

    # --- 3. 逻辑分支 ---
    if not is_leaking:
        # 情况 A：气泡捕获，使用固定 25px 闭运算
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        bubble_mask = cv2.morphologyEx(bubble_mask, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        x, y, rw, rh = cv2.boundingRect(c)
        pad = 10
    else:
        # 情况 B：边缘雷达增强模式
        print("--- 模式：增强雷达探测 ---")
        edges = cv2.Canny(gray, 50, 150)

        # 使用更大的、且具有方向性的核（50x50），把更远的字也“粘”过来
        # 增加一个专门针对漫画排版（横竖都有可能）的闭合操作
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
        dilated = cv2.dilate(edges, kernel_large)

        r_cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        target_box = None
        for rc in r_cnts:
            rx, ry, rrw, rrh = cv2.boundingRect(rc)
            if rx <= cx <= rx + rrw and ry <= cy <= ry + rrh:
                target_box = [rx, ry, rrw, rrh]
                break

        if target_box:
            # --- 【核心改进点】：二次聚类探测 ---
            # 找到包含点击点的块后，在它附近（比如 60 像素内）再找找有没有其他块
            # 这能防止“只识别一个字”的情况
            tx, ty, tw, th = target_box
            expansion_pixel = 60  # 扩张查找范围
            final_x1, final_y1 = tx, ty
            final_x2, final_y2 = tx + tw, ty + th

            for rc in r_cnts:
                rx, ry, rrw, rrh = cv2.boundingRect(rc)
                # 如果这个块离我们目标块很近，就把它并进来
                if (abs(rx - (tx + tw)) < expansion_pixel or abs(tx - (rx + rrw)) < expansion_pixel) and \
                        (abs(ry - ty) < 100 or abs(ty - ry) < 100):
                    final_x1 = min(final_x1, rx)
                    final_y1 = min(final_y1, ry)
                    final_x2 = max(final_x2, rx + rrw)
                    final_y2 = max(final_y2, ry + rrh)

            x, y, rw, rh = final_x1, final_y1, final_x2 - final_x1, final_y2 - final_y1
            pad = 20
        else:
            # 最终保底：返回中心区域
            x, y, rw, rh, pad = cx - 150, cy - 100, 300, 200, 0

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

    # 如果想看雷达探测时的“胶水”粘连效果，可以保存这个
    if is_leaking:
        # 这里假设 dilated 是你在雷达模式下生成的变量
        cv2.imwrite("debug_radar_mask.png", dilated)
    else:
        cv2.imwrite("debug_bubble_mask.png", bubble_mask)
    # --- 调试代码结束 ---

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


def get_ai_translation(text: str):
    if not text.strip():
        return ""

    try:
        start_time = time.time()
        # 使用极简 Prompt：不要求解释，只要求地道翻译和核心词原型
        # noinspection PyTypeChecker
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个漫画翻译器，直接返回译文。若有生僻动词，在译文后括号内标注[原型]"},
                {"role": "user", "content": text},
            ],
            stream=False,
            temperature=0.3,  # 降低随机性，让翻译更稳定
            max_tokens=150  # 限制输出长度，减少传输耗时
        )
        duration = time.time() - start_time
        print(f"AI 响应耗时: {duration:.2f}s")
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"翻译出错了: {str(e)}"


# 缓存最近一次的 OCR 文本，方便第二个请求直接获取
last_ocr_text = ""
@app.post("/ocr")
async def perform_ocr(payload: dict = Body(...)):
    global last_ocr_text  # 必须声明：我要修改的是外面那个全局变量
    last_ocr_text = ""  # 每次识别新图前先清空旧缓存
    img_b64 = payload.get("image")
    # 获取 Android 传来的点击坐标
    click_x = payload.get("x", 0)
    click_y = payload.get("y", 0)
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
    global last_ocr_text  # 读取全局变量
    if not last_ocr_text:
        return {"translation": "未检测到待翻译文字"}

    # 这里调用你之前的 DeepSeek 翻译函数
    translation = get_ai_translation(last_ocr_text)
    return {"translation": translation}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=12233)