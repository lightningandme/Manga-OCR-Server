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
def get_real_model_path(hf_home_dir):
    """自动定位 snapshots 下的具体模型路径"""
    base_path = Path(hf_home_dir) / "hub/models--kha-white--manga-ocr-base/snapshots"
    if not base_path.exists():
        return None

    # 获取 snapshots 下的所有文件夹（通常只有一个长字符串命名的文件夹）
    snapshots = [d for d in base_path.iterdir() if d.is_dir()]
    if not snapshots:
        return None

    # 返回最新的一个快照路径
    return str(snapshots[0])

# --- 配置区 ---
#BASE_URL = "http://192.168.137.1:4567/api/v1/manga/49"
BASE_URL = "http://10.0.0.2:2333/api/v1/manga/3557"
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
    print(f"正在为 {target_dir} 生成 OCR 数据...")

    # 核心：获取绝对物理路径
    real_path = get_real_model_path(os.environ["HF_HOME"])

    if not real_path:
        print("错误：在 HF_HOME 中未找到离线模型，请检查路径。")
        return

    print(f"检测到物理模型路径: {real_path}")

    cmd = [
        sys.executable, "-m", "mokuro",
        "--disable_confirmation",
        "--ignore_errors",
        "--pretrained_model_name_or_path", real_path,  # 传入绝对路径
        str(target_dir)
    ]

    # 建议加上 env 参数，确保环境变量 100% 传递给子进程
    current_env = os.environ.copy()

    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=current_env)

    if result.returncode == 0 and "ERROR" not in result.stderr:
        print("Mokuro 处理完成。")
    else:
        print(f"Mokuro 运行失败。错误详情:\n{result.stderr}")


def extract_script(target_dir):
    target_path = Path(target_dir)
    # 获取同级的 .mokuro 文件
    mokuro_file = target_path.with_suffix('.mokuro')

    print(f"正在分析数据文件: {mokuro_file}")

    if not mokuro_file.exists():
        print(f"未找到数据文件: {mokuro_file}")
        return

    try:
        with open(mokuro_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        script_path = target_path / "script.txt"
        with open(script_path, 'w', encoding='utf-8') as f_out:
            # 写入漫画标题背景信息
            f_out.write(f"作品名: {data.get('title', '未知')}\n")
            f_out.write(f"章节: {data.get('volume', '未知')}\n")
            f_out.write("=" * 30 + "\n")

            for i, page in enumerate(data.get('pages', [])):
                img_name = page.get('img_path', f'Page {i}')
                f_out.write(f"\n[ 第 {i + 1} 页 ({img_name}) ]\n")

                # 遍历每一个气泡块
                for block in page.get('blocks', []):
                    # 获取 lines 列表并合并
                    lines = block.get('lines', [])
                    if lines:
                        # 漫画中一个 block 里的多行通常是一句话，直接拼接
                        full_line = "".join(lines)
                        f_out.write(f"{full_line}\n")

        print(f"台本提取成功，保存在: {script_path}")

    except Exception as e:
        print(f"提取脚本时发生错误: {e}")


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