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
        if main_pos in ['記号', '助詞', '助動詞'] and token.surface in [' ', '　', '。', '、']:
            continue

        results.append({
            "s": token.surface,  # 表面形：漫画里显示的原文
            "b": token.base_form,  # 原型：查词典用的原始形态
            "p": main_pos,  # 词性：名词、动词、形容词等
            "r": token.reading  # 读音：片假名读音 (可选)
        })
    return results


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

        # 2. 本地分词 (耗时通常 < 20ms)
        words = analyze_text(text)

        print(f"OCR: {text}")
        print(f"Words: {[w['s'] for w in words]}")

        # 3. 返回组合数据
        return {
            "status": "success",
            "text": text,  # 原始字符串
            "words": words  # 分词后的列表对象
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=12233)