import os
import sys
import json
import requests
import subprocess
from pathlib import Path
from requests.auth import HTTPBasicAuth

# --- 1. 完全复用 server.py 的路径逻辑 ---
current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent
root_dir = current_dir.parent

# 确保模块导入路径一致
for p in [current_dir, root_dir]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# --- 2. 强制离线化环境变量 ---
# 这一步至关重要！确保 mokuro 内部调用的 manga-ocr 也能找到模型
os.environ["HF_HOME"] = str(root_dir / "huggingface")

# --- 3. 配置参数 ---
# 建议将缓存目录放在 root_dir 下，方便管理
STORAGE_DIR = root_dir / "manga_cache"
STORAGE_DIR.mkdir(exist_ok=True)

# 指向你的模型路径 (mokuro 的参数 --pretrained_model_name_or_path 可以直接用这个)
# 通常 manga-ocr 的模型文件夹在 HF_HOME 下的特定位置，或者你直接指定具体目录
# MODEL_PATH = "kha-white/manga-ocr"
# MODEL_PATH = str(root_dir / "huggingface" / "hub" / "models--kha-white--manga-ocr" / "snapshots" / "...")
# 注意：上面的 MODEL_PATH 建议根据你本地实际的文件夹名微调，
# 或者直接传 "kha-white/manga-ocr"，只要 HF_HOME 设置对了，它会自动去里面找。

# --- 配置区 ---
BASE_URL = "http://192.168.137.1:4567/api/v1/manga/49"
#BASE_URL = "http://10.0.0.2:2333/api/v1/manga/3557"
AUTH = HTTPBasicAuth('guest', '123')


def download_chapter(manga_id, chapter_idx):
    """循环下载某一章节的所有页面"""
    chapter_dir = os.path.join(STORAGE_DIR, f"chapter_{chapter_idx}")
    os.makedirs(chapter_dir, exist_ok=True)

    page_idx = 0
    print(f"开始下载第 {chapter_idx} 话...")

    while True:
        # 构建请求 URL (注意：此处假设你的 API 返回的是原始图像流)
        url = f"{BASE_URL}/chapter/{chapter_idx}/page/{page_idx}"
        try:
            response = requests.get(url, auth=AUTH, timeout=10)
            if response.status_code != 200:
                print(f"章节 {chapter_idx} 结束，共下载 {page_idx} 页。")
                break

            # 保存图片 (后缀名根据实际情况，通常是 .jpg 或 .png)
            file_path = os.path.join(chapter_dir, f"{page_idx:03d}.jpg")
            with open(file_path, 'wb') as f:
                f.write(response.content)

            page_idx += 1
        except Exception as e:
            print(f"下载出错: {e}")
            break
    return chapter_dir if page_idx > 0 else None


def run_mokuro(target_dir):
    """带参数运行 mokuro"""
    print(f"正在为 {target_dir} 生成 OCR 数据...")
    cmd = [
        sys.executable, "-m", "mokuro",
        target_dir,
        "--disable_confirmation",
        "--disable_html",
        "--ignore_errors",
        "--pretrained_model_name_or_path", os.environ["HF_HOME"]
    ]
    # 打印一下实际执行的命令，方便调试
    print(f"执行命令: {' '.join(cmd)}")
    # 使用 subprocess 运行并捕获输出
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.returncode == 0:
        print("Mokuro 处理完成。")
    else:
        # 这里把错误信息打印出来，看看是不是模型路径或其他问题
        print(f"Mokuro 运行失败，错误详情:\n{result.stderr}")


def extract_script(target_dir):
    """从生成的 .mokuro 文件中提取纯台本"""
    # mokuro 通常在目录下生成一个与目录名相同的 .mokuro 文件
    dirname = os.path.basename(target_dir)
    mokuro_file = os.path.join(STORAGE_DIR, f"{dirname}.mokuro")

    if not os.path.exists(mokuro_file):
        print(f"未找到数据文件: {mokuro_file}")
        return

    with open(mokuro_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    script_path = os.path.join(target_dir, "script.txt")
    with open(script_path, 'w', encoding='utf-8') as f_out:
        for i, page in enumerate(data['pages']):
            f_out.write(f"\n--- 第 {i} 页 ---\n")
            for block in page['blocks']:
                f_out.write(f"{block['text']}\n")

    print(f"台本提取成功: {script_path}")


# --- 主循环逻辑 ---
def main():
    current_chapter = 12  # 从第 12 话开始测试

    while True:
        # 1. 下载图片
        chapter_path = download_chapter(49, current_chapter)

        if not chapter_path:
            print("所有章节处理完毕或无法获取新章节。")
            break

        # 2. OCR 处理
        run_mokuro(chapter_path)

        # 3. 提取剧本
        extract_script(chapter_path)

        # 4. 章节递增
        current_chapter += 1
        print("-" * 30)


if __name__ == "__main__":
    main()