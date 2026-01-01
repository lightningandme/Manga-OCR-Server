import sys
import os
import base64
import io
from PIL import Image
from fastapi import FastAPI, Body
import uvicorn

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
        response = client.chat.completions.create(
            model="deepseek-chat",  # V3 模型，极速响应
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

@app.post("/ocr")
async def perform_ocr(payload: dict = Body(...)):
    img_b64 = payload.get("image")
    if not img_b64:
        return {"status": "error", "message": "No image data"}

    try:
        # 1. 解码与识别
        img_data = base64.b64decode(img_b64)
        image = Image.open(io.BytesIO(img_data))
        text = mocr(image)

        # 2. 调用 AI 翻译 (云端，约 1s)
        translation = get_ai_translation(text)

        # 3. 本地分词 (本地，毫秒级)
        words = analyze_text(text)

        return {
            "status": "success",
            "text": text,
            "translation": translation,
            "words": words
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}`


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=12233)