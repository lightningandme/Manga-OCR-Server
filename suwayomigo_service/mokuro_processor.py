import os
import sys
import sqlite3
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
DB_PATH = root_dir / "manga_database.db"

# --- 2.1 数据库管理 ---

def init_db():
    """初始化 SQLite 数据库"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS manga_lines (
            manga_name TEXT,
            manga_id INTEGER,
            chapter_idx INTEGER,
            page_idx TEXT,
            line_idx TEXT,
            img_width INTEGER,
            img_height INTEGER,
            box TEXT,
            content TEXT,
            translation TEXT,
            PRIMARY KEY (manga_id, chapter_idx, page_idx, line_idx)
        )
    ''')
    conn.commit()
    conn.close()

def save_script_to_db(manga_name, manga_id, chapter_idx, parsing_data):
    """
    保存解析后的脚本数据到数据库
    parsing_data: list of dict, keys: page_idx, line_idx, img_width, img_height, box, content
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for item in parsing_data:
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO manga_lines 
                (manga_name, manga_id, chapter_idx, page_idx, line_idx, img_width, img_height, box, content, translation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT translation FROM manga_lines WHERE manga_id=? AND chapter_idx=? AND page_idx=? AND line_idx=?), NULL))
            ''', (
                manga_name, manga_id, chapter_idx,
                item['page_idx'], item['line_idx'],
                item['img_width'], item['img_height'],
                item['box'], item['content'],
                manga_id, chapter_idx, item['page_idx'], item['line_idx']
            ))
        except Exception as e:
            print(f"DB Insert Error: {e}")
            
    conn.commit()
    conn.close()

def update_translation_in_db(manga_id, chapter_idx, trans_map):
    """
    更新数据库中的翻译字段
    trans_map: dict { "PageXXX_LineXXX": "Translation" }
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for key, trans_text in trans_map.items():
        try:
            # key format: Page001_Line001
            parts = key.split('_')
            if len(parts) >= 2:
                p_idx = parts[0]
                l_idx = parts[1]
                cursor.execute('''
                    UPDATE manga_lines SET translation = ?
                    WHERE manga_id = ? AND chapter_idx = ? AND page_idx = ? AND line_idx = ?
                ''', (trans_text, manga_id, chapter_idx, p_idx, l_idx))
        except Exception as e:
            print(f"DB Update Error: {e}")
            
    conn.commit()
    conn.close()


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

def download_single_page(base_url, auth_user, auth_pass, manga_name, manga_id, chapter_idx, page_idx):
    """
    尝试下载单张图片
    :param auth_user: 传入用户名
    :param auth_pass: 传入密码
    """
    path = STORAGE_ROOT / str(manga_name) / str(manga_id) / str(chapter_idx)
    path.mkdir(parents=True, exist_ok=True)

    if (path / "script.txt").exists():
        return 0

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


def generate_script_file(target_dir, manga_name, manga_id, chapter_idx):
    """
    解析 .mokuro 生成全维度结构化脚本
    格式: manga_name,manga_id,chapter_idx,PageXXX,LineXXX,Width,Height,[box],Content
    """
    target_path = Path(target_dir)
    # 自动定位 .mokuro 文件
    mokuro_filename = target_path.name + ".mokuro"
    mokuro_file = target_path.parent / mokuro_filename

    if not mokuro_file.exists():
        mokuro_file = target_path / mokuro_filename

    if not mokuro_file.exists():
        candidates = list(target_path.parent.glob("*.mokuro")) + list(target_path.glob("*.mokuro"))
        if candidates:
            mokuro_file = candidates[0]
        else:
            print(f"找不到 OCR 数据，跳过脚本生成。")
            return

    try:
        with open(mokuro_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        script_path = target_path / "script.txt"

        with open(script_path, 'w', encoding='utf-8') as f_out:
            # 按文件名排序页面
            pages = sorted(data.get('pages', []), key=lambda x: x.get('img_path', ''))

            for p_idx, page in enumerate(pages):
                img_w = page.get('img_width', 0)
                img_h = page.get('img_height', 0)
                page_num_str = f"{p_idx + 1:03d}"

                blocks = page.get('blocks', [])
                for b_idx, block in enumerate(blocks):
                    line_num_str = f"{b_idx + 1:03d}"
                    box = block.get('box', [0, 0, 0, 0])

                    # 合并文本并清洗掉破坏 CSV 结构的字符
                    content = "".join(block.get('lines', [])).replace('\n', '').replace(',', '，')

                    # 组合成全维度数据行
                    # 建议对 manga_name 也做一次逗号替换，防止名字里带逗号
                    safe_manga_name = str(manga_name).replace(',', '，')

                    output_line = (
                        f"{safe_manga_name},{manga_id},{chapter_idx},"
                        f"Page{page_num_str},Line{line_num_str},"
                        f"{img_w},{img_h},{box},{content}\n"
                    )
                    f_out.write(output_line)

                    # 收集数据用于数据库
                    db_entry = {
                        'page_idx': f"Page{page_num_str}",
                        'line_idx': f"Line{line_num_str}",
                        'img_width': img_w,
                        'img_height': img_h,
                        'box': json.dumps(box),
                        'content': content
                    }
                    save_script_to_db(manga_name, manga_id, chapter_idx, [db_entry])

        print(f"结构化脚本(含元数据)已更新: {script_path}")
        
        # 删除同目录下的图片文件
        for img_file in target_path.glob("*.jpg"):
            try:
                img_file.unlink()
            except OSError as e:
                print(f"删除图片失败 {img_file}: {e}")
        print(f"已清理目录下的图片文件。")

    except Exception as e:
        print(f"生成脚本失败: {e}")


def translate_full_chapter_to_json(ai_client, your_model, manga_name, manga_id, chapter_idx, script_lines):
    """
    接收外部传入的 ai_client 进行批量翻译
    """
    if not ai_client:
        print("跳过翻译：AI 客户端未初始化")
        return []

    if not script_lines:
        return []

    # 准备待翻译的纯文本列表，带上索引以便 AI 对应
    # 格式：{"id": "Page001_Line001", "text": "原文"}
    translate_payload = []
    for line in script_lines:
        parts = line.strip().split(',', 8)
        if len(parts) < 9: continue
        translate_payload.append({
            "id": f"{parts[3]}_{parts[4]}",
            "text": parts[8]
        })

    # 构建 Prompt
    system_content = (
        f"你是一位精通多门语言的日本漫画翻译专家，正在翻译《{manga_name}》第{chapter_idx}话。\n"
        "我会给你一个包含多个 ID 和文本的 JSON 列表。请你完成以下任务：\n"
        "1. 校对并修正 OCR 识别错误。\n"
        "2. 结合前后文，将文本翻译成地道、流畅的中文。\n"
        "3. **强制返回 JSON 数组格式**，数组中的每个对象必须包含原有的 'id' 和翻译后的 'trans' 字段。\n"
        "注意：不要返回任何解释文字，只返回 JSON 代码块。"
    )

    try:
        response = ai_client.chat.completions.create(
            model=your_model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": json.dumps(translate_payload, ensure_ascii=False)}
            ],
            stream=False,
            timeout=60.0,
            response_format={"type": "json_object"},
            temperature=0.3
        )

        raw_res = response.choices[0].message.content.strip()

        # 移除 Markdown 代码块包裹
        if raw_res.startswith("```"):
            raw_res = raw_res.split("\n", 1)[-1].rsplit("\n", 1)[0].strip()
        if raw_res.startswith("json"):  # 处理 ```json 这种开头
            raw_res = raw_res[4:].strip()

        res_data = json.loads(raw_res)

        # --- 健壮的解析逻辑 ---
        # --- 健壮的解析逻辑 ---
        final_list = []
        if isinstance(res_data, list):
            final_list = res_data
        elif isinstance(res_data, dict):
            # 优先找 translations 键，找不到就看字典里有没有唯一的 list
            if "translations" in res_data:
                final_list = res_data["translations"]
            else:
                # 自动寻找字典中任何是列表类型的字段
                for val in res_data.values():
                    if isinstance(val, list):
                        final_list = val
                        break
        
        if not final_list:
            print(f"⚠️ AI 返回了非预期格式: {raw_res}")
            return []

        # 获取目录路径
        path = STORAGE_ROOT / str(manga_name) / str(manga_id) / str(chapter_idx)
        script_zh_path = path / "script_zh.txt"
        
        # 生成 script_zh.txt 并写入 DB
        trans_map = {item['id']: item.get('trans', '') for item in final_list}
        
        try:
            with open(script_zh_path, 'w', encoding='utf-8') as f_zh:
                for line in script_lines:
                    parts = line.strip().split(',', 8)
                    if len(parts) < 9: 
                        continue
                    
                    # 构造 ID 匹配
                    line_id = f"{parts[3]}_{parts[4]}" # PageXXX_LineXXX
                    trans_text = trans_map.get(line_id, "")
                    
                    # 替换最后一部分 Content 为译文 (保持 CSV 格式)
                    # 格式: manga_name,manga_id,chapter_idx,PageXXX,LineXXX,Width,Height,[box],TransContent
                    new_parts = parts[:-1] + [trans_text]
                    f_zh.write(",".join(str(p) for p in new_parts) + "\n")
            
            print(f"已生成译文脚本: {script_zh_path}")
            
            # 更新数据库
            update_translation_in_db(manga_id, chapter_idx, trans_map)
            
        except Exception as e:
            print(f"写入译文脚本或数据库失败: {e}")

        return final_list

    except Exception as e:
        print(f"全话批量翻译失败: {e}")
        # 这里建议打印出 raw_res，看看 AI 到底吐了什么脏数据
        if 'raw_res' in locals():
            print(f"AI 原始输出内容: {raw_res}")
        return []


# --- 4. 业务逻辑控制 ---

# --- 修改后的 process_preload_request 函数定义 ---
def process_preload_request(base_url, auth_user, auth_pass, ai_client, your_model, manga_name, manga_id, start_chapter, start_page,
                            preload_count=100):
    """
    处理预读请求的主入口
    :param manga_name: 传入漫画名称，用于写入 script.txt 的每一行记录
    """
    print(f"预读启动: {manga_name} (ID: {manga_id}) | Auth: {auth_user}")

    current_chap = int(start_chapter)
    current_page = int(start_page)
    pages_left = preload_count
    affected_chapters = set()

    # --- 阶段一：流式下载 (保持之前的逻辑) ---
    # --- 阶段一：流式下载 (改进版逻辑) ---
    while True:
        # 退出条件检查：如果 page 已经读完 (pages_left <= 0)，且章节已经前进了 2 章以上
        if pages_left <= 0 and (current_chap >= int(start_chapter) + 2):
            break
            
        status = download_single_page(base_url, auth_user, auth_pass, manga_name, manga_id, current_chap, current_page)

        if status == 0:
            affected_chapters.add(current_chap)
            current_page += 1
            if pages_left > 0:
                pages_left -= 1
        elif status == 1:
            # 404 Not Found -> 这一话可能结束了，去下一话
            current_chap += 1
            current_page = 0
            # 安全熔断：如果超过起始章节太多(比如10话)都找不到，还是得停，防止无限试错
            if current_chap > int(start_chapter) + 10: 
                break
        else:
            # 其他错误 (status=2)
            break

    # --- 阶段二：批量 OCR 和 剧本转化 (此处必须修改) ---
    for chap_idx in affected_chapters:
        chap_dir = STORAGE_ROOT / str(manga_name) / str(manga_id) / str(chap_idx)

        # 1. 运行 OCR
        run_mokuro_on_dir(chap_dir)

        # 2. 生成结构化剧本，传入新增的 manga_name
        # 注意：这里的参数顺序必须与 generate_script_file 定义的一致
        generate_script_file(chap_dir, manga_name, manga_id, chap_idx)

        # 2. 读取刚生成的脚本
        script_file = chap_dir / "script.txt"
        with open(script_file, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()

        # 3. 启动全话 AI 翻译 (JSON 模式)
        # 3. 启动全话 AI 翻译 (JSON 模式)
        if ai_client:
            print(f"--- 正在通过 AI JSON 模式翻译全话: {manga_name} 第 {chap_idx} 话 ---")
            # 注意: 这里 update 了参数，传入 manga_id
            translate_full_chapter_to_json(ai_client, your_model, manga_name, manga_id, chap_idx, original_lines)

        # (原有的 save_batch_to_db_v2 调用已移除，逻辑已整合进 translate 函数)

    print(f"[{manga_name}] 的预读任务完成。")


# --- 模拟调用示例 ---
if __name__ == "__main__":
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()  # 加载 .env 文件里的 API_KEY 等
    except ImportError:
        print("请确保已安装 openai 和 python-dotenv 库")
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    your_model = os.getenv("YOUR_MODEL")

    process_preload_request(
        base_url="http://192.168.137.1:4567/api/v1",
        auth_user="guest",
        auth_pass="123",
        ai_client=OpenAI(api_key=api_key, base_url=base_url),
        your_model=your_model,
        manga_name="ruri_dragon",
        manga_id=49,
        start_chapter=12,
        start_page=0
    )