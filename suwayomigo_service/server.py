import sys
import os

# 关键：将项目根目录加入系统路径，确保能找到 manga_ocr 文件夹
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Body
from manga_ocr import MangaOcr  # 直接调用原项目的类
import base64
import io
from PIL import Image

app = FastAPI()

# 全局初始化一次模型，避免重复加载
# 如果你有多显卡或想指定 CPU，可以在这里配置参数
mocr = MangaOcr()


@app.post("/ocr")
async def perform_ocr(payload: dict = Body(...)):
    img_b64 = payload.get("image")
    if not img_b64:
        return {"status": "error", "message": "No image data"}

    # 解码图片
    img_data = base64.b64decode(img_b64)
    image = Image.open(io.BytesIO(img_data))

    # 调用 MangaOcr 原本的识别方法
    text = mocr(image)

    print(f"识别结果: {text}")
    return {"status": "success", "text": text}


if __name__ == "__main__":
    import uvicorn

    # host="0.0.0.0" 才能让手机访问
    uvicorn.run(app, host="0.0.0.0", port=12233)