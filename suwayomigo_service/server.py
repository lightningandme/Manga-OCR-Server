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

def get_smart_crop(image_bytes, click_x, click_y):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None
    h, w = img.shape[:2]
    # 【调试】在图上画一个红点，看看点击位置对不对
    #debug_img = img.copy()
    #cv2.circle(debug_img, (click_x, click_y), 5, (0, 0, 255), -1)
    #cv2.imwrite("debug_click_point.png", debug_img)

    # 1. 预处理：灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 自动纠偏逻辑
    # 显式将坐标转为 int，防止从 JSON 传来的是 float
    cx, cy = int(click_x), int(click_y)

    if gray[cy, cx] < 150:
        radius = 15
        min_y, max_y = max(0, cy - radius), min(h, cy + radius)
        min_x, max_x = max(0, cx - radius), min(w, cx + radius)
        sub_region = gray[min_y:max_y, min_x:max_x]

        if sub_region.size > 0:
            _, max_val, _, max_loc = cv2.minMaxLoc(sub_region)
            if max_val > 180:
                cx = min_x + max_loc[0]
                cy = min_y + max_loc[1]
                print(f"--- 自动纠偏成功至: ({cx}, {cy}) ---")

    # 3. 魔法棒算法 (FloodFill)
    # 【修复重点】：mask 必须是 uint8，且大小为 (h+2, w+2)
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)

    # 【修复重点】：loDiff 和 upDiff 建议传入元组格式 (Scalar)
    # 即使是灰度图，也推荐使用 (value,) 或 (value, value, value)
    diff = (20,)

    # 复制一份用于填充（FloodFill 会直接修改原图）
    flood_filled = gray.copy()

    cv2.floodFill(
        image=flood_filled,
        mask=ff_mask,
        seedPoint=(cx, cy),
        newVal=255,
        loDiff=diff,
        upDiff=diff,
        flags=cv2.FLOODFILL_FIXED_RANGE
    )

    # 提取生成的 mask（去掉外围的 2 像素边缘）
    bubble_mask = ff_mask[1:-1, 1:-1] * 255

    # 4. 闭运算 + 空洞填充
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    bubble_mask = cv2.morphologyEx(bubble_mask, cv2.MORPH_CLOSE, kernel)

    # 填充所有闭合轮廓内部
    cnts, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        cv2.drawContours(bubble_mask, [c], 0, 255, -1)

    # 5. 寻找并切图
    if cnts:
        for c in cnts:
            x, y, rw, rh = cv2.boundingRect(c)
            # 再次确认点在轮廓内
            if x <= cx <= x + rw and y <= cy <= y + rh:
                if rw < w * 0.98:
                    x_n, y_n = max(0, x - 15), max(0, y - 15)
                    w_n, h_n = min(w - x_n, rw + 30), min(h - y_n, rh + 30)
                    return img[y_n:y_n + h_n, x_n:x_n + w_n]

    return img

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