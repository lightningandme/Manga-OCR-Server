import os
import sys
import json
import requests
import subprocess
from pathlib import Path
from requests.auth import HTTPBasicAuth
import time

# --- 1. 路径与环境初始化 (保持不变) ---
current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent
root_dir = current_dir.parent

for p in [current_dir, root_dir]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ["HF_HOME"] = str(root_dir / "huggingface")

# --- 2. 配置参数 ---
# 修改缓存根目录结构，将在内部按照 {manga_id}/{chapter_idx} 划分
STORAGE_ROOT = root_dir / "manga_cache"
STORAGE_ROOT.mkdir(exist_ok=True)

# 基础 URL 模板 (注意去掉了具体的 manga ID，将在调用时动态拼接)
BASE_API_URL = "http://10.0.0.2:2333/api/v1"
AUTH = HTTPBasicAuth('guest', '123')


def get_real_model_path(hf_home_dir):
    """自动定位 snapshots 下的具体模型路径 (保持不变)"""
    base_path = Path(hf_home_dir) / "hub/models--kha-white--manga-ocr-base/snapshots"
    if not base_path.exists():
        return None
    snapshots = [d for d in base_path.iterdir() if d.is_dir()]
    if not snapshots:
        return None
    return str(snapshots[0])


# --- 3. 核心功能函数 ---

def get_chapter_dir(manga_id, chapter_idx):
    """生成标准化的章节存储路径: manga_cache/3557/12"""
    # 路径结构: manga_cache / manga_id / chapter_idx
    path = STORAGE_ROOT / str(manga_id) / str(chapter_idx)
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_single_page(manga_id, chapter_idx, page_idx):
    """
    尝试下载单张图片
    返回状态:
    0 = 成功 (或已存在)
    1 = 章节结束 (404)
    2 = 其他错误
    """
    chapter_dir = get_chapter_dir(manga_id, chapter_idx)
    file_name = f"{page_idx:03d}.jpg"  # 统一命名格式
    file_path = chapter_dir / file_name

    # 如果文件已存在，直接视为成功，跳过下载节省流量
    if file_path.exists():
        # print(f"[{manga_id}-{chapter_idx}-{page_idx}] 本地已存在，跳过下载。")
        return 0

    url = f"{BASE_API_URL}/manga/{manga_id}/chapter/{chapter_idx}/page/{page_idx}"
    try:
        response = requests.get(url, auth=AUTH, timeout=10)

        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"下载成功: {chapter_idx}话 - P{page_idx}")
            return 0
        elif response.status_code == 404:
            # print(f"章节 {chapter_idx} 似乎在 P{page_idx} 结束了。")
            return 1
        else:
            print(f"下载异常: {response.status_code}")
            return 2
    except Exception as e:
        print(f"请求错误: {e}")
        return 2


def run_mokuro_on_dir(target_dir):
    """对指定目录运行 Mokuro (增量更新模式)"""
    print(f"正在更新 OCR 数据: {target_dir}")
    real_path = get_real_model_path(os.environ["HF_HOME"])

    if not real_path:
        print("错误：找不到离线模型路径。")
        return

    cmd = [
        sys.executable, "-m", "mokuro",
        "--disable_confirmation",
        "--ignore_errors",
        "--pretrained_model_name_or_path", real_path,
        str(target_dir)
    ]

    current_env = os.environ.copy()
    # 捕获输出以免刷屏，只在错误时显示
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=current_env)

    if result.returncode != 0:
        print(f"Mokuro 运行警告:\n{result.stderr}")


def generate_script_file(target_dir, manga_id, chapter_idx):
    """
    从 .mokuro 生成/更新 script.txt
    注意：这里采用'重写'模式。因为mokuro生成的JSON包含该文件夹下所有图片的数据，
    所以重写 script.txt 等同于'追加了新页面的内容'，且能保证顺序不乱。
    """
    target_path = Path(target_dir)
    # mokuro 文件通常生成在同级目录，文件名为 文件夹名.mokuro
    # 例如目录是 .../12，文件则是 .../12.mokuro
    mokuro_filename = target_path.name + ".mokuro"
    mokuro_file = target_path.parent / mokuro_filename  # 注意这里mokuro生成的位置机制

    # 兼容性修正：有时候mokuro会生成在目录里面，有时候在外面，取决于版本
    # 如果外面没找到，找一下里面
    if not mokuro_file.exists():
        mokuro_file = target_path / mokuro_filename

    if not mokuro_file.exists():
        # 再次尝试：直接找后缀
        candidates = list(target_path.parent.glob("*.mokuro")) + list(target_path.glob("*.mokuro"))
        if candidates:
            mokuro_file = candidates[0]
        else:
            print(f"未找到对应的OCR数据文件，跳过脚本生成。")
            return

    try:
        with open(mokuro_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        script_path = target_path / "script.txt"

        with open(script_path, 'w', encoding='utf-8') as f_out:
            f_out.write(f"MangaID: {manga_id} | Chapter: {chapter_idx}\n")
            f_out.write("=" * 30 + "\n")

            # 按照文件名排序页面，防止追加下载时顺序错乱
            pages = data.get('pages', [])
            # 简单的按 img_path 排序 (000.jpg, 001.jpg)
            pages.sort(key=lambda x: x.get('img_path', ''))

            for page in pages:
                img_name = page.get('img_path', 'Unknown')
                f_out.write(f"\n[Page: {img_name}]\n")

                for block in page.get('blocks', []):
                    lines = block.get('lines', [])
                    if lines:
                        f_out.write("".join(lines) + "\n")

        print(f"剧本已更新: {script_path}")

    except Exception as e:
        print(f"生成脚本失败: {e}")


# --- 4. 业务逻辑控制 (The Pipeline) ---

def process_preload_request(manga_id, start_chapter, start_page, preload_count=10):
    """
    处理预读请求的主入口
    :param preload_count: 预读总页数，默认为 10
    """
    print(f"收到预读请求: Manga {manga_id}, Chap {start_chapter}, Page {start_page} (往后 {preload_count} 页)")

    current_chap = int(start_chapter)
    current_page = int(start_page)
    pages_left = preload_count

    # 记录哪些章节的数据发生了变动，需要重新OCR
    affected_chapters = set()

    # --- 阶段一：流式下载 (Streaming Download) ---
    while pages_left > 0:
        status = download_single_page(manga_id, current_chap, current_page)

        if status == 0:  # 成功或已存在
            affected_chapters.add(current_chap)
            current_page += 1
            pages_left -= 1

        elif status == 1:  # 本章结束 (404)
            print(f"第 {current_chap} 话已完结，切换下一话...")
            current_chap += 1
            current_page = 0  # 重置页码
            # 注意：不扣除 pages_left，因为这一轮没下载到东西，需要在下一话补上

            # 安全中断：防止无限循环查找不存在的漫画结尾
            if current_chap > int(start_chapter) + 5:
                print("连续多话未找到内容，停止预读。")
                break

        else:  # 其他错误
            print("下载遇到不可恢复错误，停止预读。")
            break

    # --- 阶段二：批量处理 (Batch Processing) ---
    # 我们下载了多个页面，可能跨越了两个章节。
    # 为了节省显存加载时间，我们对涉及到的每个章节目录运行一次 Mokuro

    for chap_idx in affected_chapters:
        chap_dir = get_chapter_dir(manga_id, chap_idx)

        # 1. 运行 OCR (Mokuro 会自动跳过已识别的，只处理新图)
        run_mokuro_on_dir(chap_dir)

        # 2. 生成/更新 Script (包含该目录下所有图片的最新文本)
        generate_script_file(chap_dir, manga_id, chap_idx)

    print(f"预读任务完成。")


# --- 模拟入口 (用于测试) ---
if __name__ == "__main__":
    # 模拟用户看到了: 漫画ID 3557, 第 12 话, 第 5 页
    # 程序应该会自动处理 P5, P6, P7...
    # 如果 P17 是最后一页，它会自动去下载 第 13 话 的 P0, P1...

    target_manga_id = 3557
    target_chapter = 12
    target_page = 5  # 从中间开始看

    process_preload_request(target_manga_id, target_chapter, target_page)