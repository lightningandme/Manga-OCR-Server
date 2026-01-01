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
    if not img_b64:
        return {"status": "error", "message": "No image data"}

    try:
        # 1. 解码与识别
        img_data = base64.b64decode(img_b64)
        image = Image.open(io.BytesIO(img_data))
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