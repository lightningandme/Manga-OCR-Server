print("OCR服务器启动中...")
import base64
import os
import sys
import cv2
import uvicorn
from PIL import Image
from fastapi import FastAPI, Body, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
import socket

# 1. 环境与路径初始化
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 离线化关键：指定模型存放位置
os.environ["HF_HOME"] = os.path.join(current_dir, "huggingface")

from manga_ocr import MangaOcr
from janome.tokenizer import Tokenizer
from crop_engine import MangaCropEngine
import easyocr
import torch

from openai import OpenAI
import time
from deep_translator import GoogleTranslator

import warnings
# 屏蔽掉来自 huggingface_hub 的 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# 检查 GPU 是否可用 (Check GPU Availability)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- 运行环境检查 ---")
print(f"检测到可用设备: {device.upper()}")
if device == "cuda":
    print(f"显卡型号: {torch.cuda.get_device_name(0)}")
else:
    print("提示: 未检测到 NVIDIA GPU 或 CUDA 驱动，将使用 CPU 运行（速度较慢）。")

# 加载当前目录下的 .env 文件
load_dotenv()

# 从环境变量中读取
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
your_model = os.getenv("YOUR_MODEL")
ocr_secret_key = os.getenv("OCR_SECRET_KEY")
# --- 核心修改：增加 AI 可用性检测 ---
is_ai_available = False
client = None

if api_key and base_url:
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        # 这里不进行实际请求，只检查配置是否存在
        is_ai_available = True
        print("✅ AI 配置加载成功")
    except Exception as e:
        print(f"⚠️ AI 初始化失败，将启用网络翻译模式: {e}")
else:
    print("ℹ️ 未检测到 API_KEY，已自动进入网络翻译模式（如果想体验更好的AI翻译，请根据.env.example进行配置）")

app = FastAPI()

# 初始化检测器 (只开启检测功能，不开启识别，速度极快)
print("初始化 easyocr 文本检测器...")
gpu_available = torch.cuda.is_available()
reader = easyocr.Reader(['ja', 'en'], gpu=gpu_available)
crop_engine = MangaCropEngine(reader)

# 初始化 Janome 分词器 (本地运行，极快)
print("初始化 janome 分词器...")
tokenizer = Tokenizer()

# 初始化 OCR 模型
print("正在加载 Manga-OCR 模型...")
time.sleep(1)
mocr = MangaOcr()

# 从 dict_engine.py 文件中导入 dict_engine 实例
try:
    from dict_engine import dict_engine
except ImportError:
    print("❌ 无法导入 dict_engine，请确保 dict_engine.py 存在于当前目录")
    dict_engine = None

# 1. 定义 Header 名称，通常习惯用 X-API-Key
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# 2. 校验函数 (Validation Function)
async def verify_api_key(api_key: str = Security(api_key_header)):
    # 将从 Header 获取的 key 与你环境变量中的 key 比对
    if api_key == ocr_secret_key:
        return api_key
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key. Access Denied."
        )

def analyze_text(text: str):
    """
    对日语文本进行分词，并筛选出具有学习价值的词汇 (Filtering for learning value)
    """
    results = []
    # 运行分词 (Run tokenization)
    tokens = tokenizer.tokenize(text)

    # 定义我们要保留的核心词性白名单
    # 名词, 动词, 形容词, 副词, 连体词(如'この'), 感叹词(拟声拟态词)
    ALLOWED_POS = ['名詞', '動詞', '形容詞', '副詞', '連体詞', '感動詞']

    for token in tokens:
        pos_details = token.part_of_speech.split(',')
        if pos_details[0] not in ALLOWED_POS:
            continue

        base_form = token.base_form
        # 安全调用
        dict_data = {"r": "", "p": "", "d": ""}
        if dict_engine:
            dict_data = dict_engine.lookup(base_form)

        results.append({
            "s": token.surface,
            "b": base_form,
            "p": pos_details[0],
            # 优先使用 Yomichan 的精准读音
            "r": dict_data["r"] if dict_data["r"] else token.reading,
            "d": dict_data["d"]
        })

    # 去重处理：如果同一句话里同一个词出现多次，只保留一个原型
    unique_results = []
    seen_bases = set()
    for item in results:
        if item["b"] not in seen_bases:
            unique_results.append(item)
            seen_bases.add(item["b"])

    return unique_results


def get_ai_translation(text: str, manga_name: str):
    manga, episode = manga_name.rsplit(':', 1) if ':' in manga_name else ("日本漫画","某一话")
    global your_model

    if not text.strip():
        return ""

    # 1. 优先尝试 AI 翻译
    if is_ai_available and client:
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
                model=your_model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": text},
                ],
                stream=False,
                timeout=5.0,
                temperature=0.3,  # 降低随机性，让翻译更稳定
                max_tokens=150  # 限制输出长度，减少传输耗时
            )
            duration = time.time() - start_time
            print(f"AI翻译 响应耗时: {duration:.2f}s (正在看:《{manga}》的{episode})")
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"翻译出错了: {str(e)}"
    # 2. 如果 AI 失败或未配置，进入 Fallback
    return get_fallback_translation(text)

def get_fallback_translation(text: str):
    """
    网络翻译备份逻辑：多引擎重试
    """

    try:
        # 方案 A: 使用 Google 翻译 (通常最准)
        # 如果在国内环境，可能需要配置 proxies
        start_time = time.time()
        result = GoogleTranslator(source='ja', target='zh-CN').translate(text)
        print(f"🌐 网络翻译(Google)耗时: {time.time() - start_time:.2f}s")
        return f"[Google翻译] {result}"
    except Exception as e:
        print(f"⚠️ Google 翻译失败: {e}，给客户端跳出提示...")
        return f"调用Google翻译失败，请检查网络环境，或推荐使用AI翻译（配置方法详见GitHub页面）"


def prepare_for_mocr(img_mat):
    """
    针对 Manga-OCR 优化的轻量预处理 (Lightweight Preprocessing)
    """
    if img_mat is None or img_mat.size == 0:
        return None

    h, w = img_mat.shape[:2]

    # --- 核心改进：引入宽度判定 (Incorporate width detection) ---
    # 定义最小理想尺寸 (Minimum ideal dimension)
    min_threshold = 128
    target_dim = 256.0

    # 只要高度或宽度任何一个太小，就进行等比例放大
    if h < min_threshold or w < min_threshold:
        # 计算缩放比例，取能让短边达到 target_dim 的那个比例
        # 这样可以确保放大后的图片，最窄的地方也有足够的细节
        scale = target_dim / min(h, w)

        # 限制最大缩放倍数，防止图片被放大到无限大导致 OOM (Out of Memory)
        scale = min(scale, 4.0)

        img_mat = cv2.resize(img_mat, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # 轻度对比度增强与压低亮度 (让文字更黑)
        # alpha=1.3 增加对比度, beta=-15 压低亮度让灰色背景变白/黑色字更深
        img_mat = cv2.convertScaleAbs(img_mat, alpha=1.3, beta=-15)

    # 2. 增加白色留白 (Padding)
    # 核心优化：防止文字贴边导致的识别失败，同时让模型更容易聚焦
    pad_h = int(img_mat.shape[0] * 0.10)  # 上下增加 10%
    pad_w = int(img_mat.shape[1] * 0.10)  # 左右增加 10%

    img_mat = cv2.copyMakeBorder(
        img_mat, pad_h, pad_h, pad_w, pad_w,
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    return img_mat

# 用户客户端验证连通性
@app.get("/health")
async def health_check(token: str = Depends(verify_api_key)):
    return {"status": "ok"}

# 缓存最近一次的 OCR 文本和漫画名
last_ocr_text = ""
last_manga_name = "General"
@app.post("/ocr")
async def perform_ocr(payload: dict = Body(...), token: str = Depends(verify_api_key) ):
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
        # 这里的 click_x/y 应该是相对于这张局部图的坐标
        start_time = time.time()
        smart_img_mat = crop_engine.get_smart_crop(img_data, click_x, click_y)
        # 针对 Manga-OCR 的预处理优化 (Preprocessing Optimization)
        optimized_mat = prepare_for_mocr(smart_img_mat)

        # 将 OpenCV 的 Mat 转回 PIL Image 给 Manga-OCR 使用
        smart_img_rgb = cv2.cvtColor(optimized_mat, cv2.COLOR_BGR2RGB)
        #cv2.imwrite("final_crop.png", optimized_mat)
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
async def get_translation(token: str = Depends(verify_api_key)):
    global last_ocr_text, last_manga_name  # <--- 声明读取这两个全局变量
    if not last_ocr_text:
        return {"translation": "未检测到待翻译文字"}

    # 调用时传入缓存的漫画名
    translation = get_ai_translation(last_ocr_text, last_manga_name)
    print(f"[原文] -->  {last_ocr_text}")
    print(f"[译文] -->  {translation}")
    return {"translation": translation}


if __name__ == "__main__":
    # 获取本机IP地址
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)) # 连接一个外部地址，不发送数据，只为获取本地IP
        local_ip = s.getsockname()[0]
        s.close()
    except Exception :
        local_ip = "127.0.0.1" # 如果获取失败，则默认为本地回环地址

    port = 12233
    print(f"🆗 OCR服务器已启动，访问地址 -->  http://{local_ip}:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
