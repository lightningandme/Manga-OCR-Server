import base64
import os
import sys
import cv2
import uvicorn
from PIL import Image
from fastapi import FastAPI, Body
from dotenv import load_dotenv

# 关键：将项目根目录加入系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from manga_ocr import MangaOcr
from janome.tokenizer import Tokenizer
from crop_engine import MangaCropEngine
import easyocr
import torch

from openai import OpenAI
import time

import warnings
# 屏蔽掉来自 huggingface_hub 的 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

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

# 初始化检测器 (只开启检测功能，不开启识别，速度极快)
print("Initializing CRAFT Text Detector...")
gpu_available = torch.cuda.is_available()
reader = easyocr.Reader(['ja', 'en'], gpu=gpu_available)
crop_engine = MangaCropEngine(reader)


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
    manga, episode = manga_name.rsplit(':', 1) if ':' in manga_name else ("日本漫画","某一话")
    if not text.strip():
        return ""

    try:
        start_time = time.time()
        # 使用极简 Prompt：不要求解释，只要求地道翻译和核心词原型
        # noinspection PyTypeChecker

        # 按照你提供的 Prompt 模板构建 System Content
        system_content = (
            f"你是一位精通多门语言的日本漫画翻译专家，正在阅读《{manga}》的{episode}。 \n"
            "你的任务是处理来自 OCR 识别的原文，并完成以下三步：\n"
            "1. **文本校对**：判断识别结果中是否存在因笔画密集导致的错别字，请结合语境将其修正（例如将错误的形近字还原为正确的词汇）。\n"
            "2. **逻辑断句**：判断因漫画排版导致的非正常连字，并进行逻辑断行或增加标点，还原角色真实的说话节奏。\n"
            "3. **地道翻译**：基于修正后的原文，结合该作品在此阶段的剧情背景和角色身份进行翻译。\n\n"
            "请翻译成地道、流畅的中文。直接返回译文。"
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
        print(f"AI翻译 响应耗时: {duration:.2f}s (正在看: 《{manga}》的{episode})")
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
        start_time = time.time()
        smart_img_mat = crop_engine.get_smart_crop(img_data, click_x, click_y)

        # 将 OpenCV 的 Mat 转回 PIL Image 给 Manga-OCR 使用
        smart_img_rgb = cv2.cvtColor(smart_img_mat, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(smart_img_rgb)
        duration = time.time() - start_time
        print(f"图片截取 响应耗时: {duration:.2f}s")

        # 后续 OCR 逻辑不变
        start_time = time.time()
        text = mocr(image)
        last_ocr_text = text  # 存入缓存

        words = analyze_text(text)
        duration = time.time() - start_time
        print(f"文本处理 响应耗时: {duration:.2f}s")

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