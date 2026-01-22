import os
import sys
import json
import requests
import subprocess
from pathlib import Path
from requests.auth import HTTPBasicAuth

# --- 1. 路径与环境初始化 ---
current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent
root_dir = current_dir.parent

for p in [current_dir, root_dir]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ["HF_HOME"] = str(root_dir / "huggingface")

# --- 2. 配置参数 ---
STORAGE_ROOT = root_dir / "manga_cache"
STORAGE_ROOT.mkdir(exist_ok=True)


def get_real_model_path(hf_home_dir):
    """自动定位 snapshots 下的具体模型路径"""
    base_path = Path(hf_home_dir) / "hub/models--kha-white--manga-ocr-base/snapshots"
    if not base_path.exists():
        return None
    snapshots = [d for d in base_path.iterdir() if d.is_dir()]
    if not snapshots:
        return None
    return str(snapshots[0])


# --- 3. 核心功能函数 ---

def download_single_page(base_url, auth_user, auth_pass, manga_id, chapter_idx, page_idx):
    """
    尝试下载单张图片
    :param auth_user: 传入用户名
    :param auth_pass: 传入密码
    """
    path = STORAGE_ROOT / str(manga_id) / str(chapter_idx)
    path.mkdir(parents=True, exist_ok=True)

    file_path = path / f"{page_idx:03d}.jpg"

    if file_path.exists():
        return 0

    clean_base_url = base_url.rstrip("/")
    url = f"{clean_base_url}/manga/{manga_id}/chapter/{chapter_idx}/page/{page_idx}"

    # 动态创建 Auth 对象
    auth = HTTPBasicAuth(auth_user, auth_pass)

    try:
        response = requests.get(url, auth=auth, timeout=10)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"下载成功: {chapter_idx}话 - P{page_idx}")
            return 0
        elif response.status_code == 404:
            return 1
        else:
            print(f"下载异常: {response.status_code}")
            return 2
    except Exception as e:
        print(f"请求错误: {e}")
        return 2


def run_mokuro_on_dir(target_dir):
    """运行 Mokuro OCR (保持不变)"""
    real_path = get_real_model_path(os.environ["HF_HOME"])
    if not real_path:
        print("错误：找不到离线模型路径。")
        return

    cmd = [
        sys.executable, "-m", "mokuro",
        "--disable_confirmation", "--ignore_errors",
        "--pretrained_model_name_or_path", real_path,
        str(target_dir)
    ]

    subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=os.environ.copy())


def generate_script_file(target_dir, manga_id, chapter_idx):
    """解析 .mokuro 生成 script.txt (保持不变)"""
    target_path = Path(target_dir)
    # 查找逻辑适配不同版本的 mokuro 输出位置
    mokuro_file = next(target_path.parent.glob(f"{target_path.name}.mokuro"), None) or \
                  next(target_path.glob("*.mokuro"), None)

    if not mokuro_file:
        return

    try:
        with open(mokuro_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        script_path = target_path / "script.txt"
        with open(script_path, 'w', encoding='utf-8') as f_out:
            f_out.write(f"MangaID: {manga_id} | Chapter: {chapter_idx}\n" + "=" * 30 + "\n")
            pages = sorted(data.get('pages', []), key=lambda x: x.get('img_path', ''))
            for page in pages:
                f_out.write(f"\n[Page: {page.get('img_path', 'Unknown')}]\n")
                for block in page.get('blocks', []):
                    f_out.write("".join(block.get('lines', [])) + "\n")
        print(f"剧本更新完毕: {script_path}")
    except Exception as e:
        print(f"脚本生成失败: {e}")


# --- 4. 业务逻辑控制 ---

def process_preload_request(base_url, auth_user, auth_pass, manga_id, start_chapter, start_page, preload_count=10):
    """
    处理预读请求的主入口
    """
    print(f"预读启动: Manga {manga_id} | Auth: {auth_user}")

    current_chap = int(start_chapter)
    current_page = int(start_page)
    pages_left = preload_count
    affected_chapters = set()

    # 阶段一：流式下载
    while pages_left > 0:
        status = download_single_page(base_url, auth_user, auth_pass, manga_id, current_chap, current_page)

        if status == 0:
            affected_chapters.add(current_chap)
            current_page += 1
            pages_left -= 1
        elif status == 1:
            current_chap += 1
            current_page = 0
            if current_chap > int(start_chapter) + 5: break
        else:
            break

    # 阶段二：批量 OCR 和 剧本转化
    for chap_idx in affected_chapters:
        chap_dir = STORAGE_ROOT / str(manga_id) / str(chap_idx)
        run_mokuro_on_dir(chap_dir)
        generate_script_file(chap_dir, manga_id, chap_idx)


# --- 模拟调用示例 ---
if __name__ == "__main__":
    process_preload_request(
        base_url="http://192.168.137.1:4567/api/v1",
        auth_user="guest",
        auth_pass="123",
        manga_id=49,
        start_chapter=12,
        start_page=1
    )