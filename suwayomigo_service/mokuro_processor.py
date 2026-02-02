import os
import sys
import sqlite3
import json
import requests
import subprocess
import time
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

    # --- 修改点：如果剧本已存在，返回特殊状态码 3 ---
    if (path / "script.txt").exists():
        # 这里不要打印，否则会刷屏
        return 3

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
    接收外部传入的 ai_client 进行批量翻译（支持分批处理以避免 Token 超限）
    """
    if not ai_client:
        print("跳过翻译：AI 客户端未初始化")
        return []

    if not script_lines:
        return []

    # 1. 准备待翻译的纯文本列表
    translate_payload = []
    for line in script_lines:
        parts = line.strip().split(',', 8)
        if len(parts) < 9: continue
        translate_payload.append({
            "id": f"{parts[3]}_{parts[4]}",  # PageXXX_LineXXX
            "text": parts[8]
        })

    if not translate_payload:
        return []

    # --- 核心修改：分批处理逻辑 ---
    BATCH_SIZE = 40  # 每批处理 40 行，防止 AI 输出截断
    final_list = []
    total_items = len(translate_payload)

    print(f"--- 开始翻译: {manga_name} 第 {chapter_idx} 话 (共 {total_items} 行) ---")

    # 构建基础 Prompt (保持不变)
    system_content = (
        f"你是一位精通多门语言的日本漫画翻译专家，正在翻译《{manga_name}》第{chapter_idx}话。\n"
        "我会给你一个包含多个 ID 和文本的 JSON 列表。请你完成以下任务：\n"
        "1. 校对并修正 OCR 识别错误。\n"
        "2. 结合前后文，将文本翻译成地道、流畅的中文。\n"
        "3. **强制返回 JSON 数组格式**，数组中的每个对象必须包含原有的 'id' 和翻译后的 'trans' 字段。\n"
        "注意：不要返回任何解释文字，只返回 JSON 代码块。"
    )

    # 循环切片
    for i in range(0, total_items, BATCH_SIZE):
        batch = translate_payload[i: i + BATCH_SIZE]
        print(
            f"  > 正在处理批次 {i // BATCH_SIZE + 1}/{(total_items + BATCH_SIZE - 1) // BATCH_SIZE} (行 {i + 1}~{min(i + BATCH_SIZE, total_items)})...")

        # 重试机制：每个批次最多重试 3 次
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = ai_client.chat.completions.create(
                    model=your_model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": json.dumps(batch, ensure_ascii=False)}
                    ],
                    stream=False,
                    timeout=90.0,  # 稍微增加超时时间
                    response_format={"type": "json_object"},
                    temperature=0.3
                )

                raw_res = response.choices[0].message.content.strip()

                # 清洗 Markdown 标记
                if raw_res.startswith("```"):
                    raw_res = raw_res.split("\n", 1)[-1].rsplit("\n", 1)[0].strip()
                if raw_res.startswith("json"):
                    raw_res = raw_res[4:].strip()

                batch_res_data = json.loads(raw_res)

                # 解析当次批次的结果
                batch_final = []
                if isinstance(batch_res_data, list):
                    batch_final = batch_res_data
                elif isinstance(batch_res_data, dict):
                    if "translations" in batch_res_data:
                        batch_final = batch_res_data["translations"]
                    else:
                        for val in batch_res_data.values():
                            if isinstance(val, list):
                                batch_final = val
                                break

                if batch_final:
                    final_list.extend(batch_final)  # 合并结果
                    break  # 成功则跳出重试循环
                else:
                    raise ValueError("解析后未找到列表数据")

            except Exception as e:
                print(f"    [警告] 批次处理失败 (第 {attempt + 1} 次重试): {e}")
                if attempt == max_retries - 1:
                    print("    [错误] 该批次翻译最终失败，跳过。")
                time.sleep(2)  # 失败后稍作等待

    # --- 后处理逻辑 (写入文件和数据库) ---
    if not final_list:
        print("⚠️ 未获取到任何翻译结果")
        return []

    print(f"全话翻译完成，共获取 {len(final_list)} 条译文")

    path = STORAGE_ROOT / str(manga_name) / str(manga_id) / str(chapter_idx)
    script_zh_path = path / "script_zh.txt"
    trans_map = {item['id']: item.get('trans', '') for item in final_list}

    try:
        with open(script_zh_path, 'w', encoding='utf-8') as f_zh:
            for line in script_lines:
                line = line.strip()
                if not line: continue

                # 1. 依然使用 rsplit 精准切分出 [元数据] 和 [原文]
                parts_fixed = line.rsplit(',', 1)
                if len(parts_fixed) < 2: continue

                meta_part = parts_fixed[0]
                original_text = parts_fixed[1]

                # 2. 仅为了获取 ID (PageXXX_LineXXX) 进行 split
                # 我们只需要前 5 个字段，后面的坐标部分我们不碰它
                temp_elements = meta_part.split(',')
                if len(temp_elements) < 5: continue
                line_id = f"{temp_elements[3]}_{temp_elements[4]}"

                # 3. 获取译文
                trans_text = trans_map.get(line_id, "")

                # 4. 写入文件：直接拼接 meta_part，不重新组合 box，确保 100% 还原
                f_zh.write(f"{meta_part},{trans_text}\n")

                # 5. 同步数据库 (修复 box 提取逻辑)
                # 找到第一个 "[" 和最后一个 "]" 的位置来提取完整的 box
                start_box = meta_part.find('[')
                end_box = meta_part.rfind(']')
                box_str = meta_part[start_box:end_box + 1] if start_box != -1 else ""

                db_entry = {
                    'page_idx': temp_elements[3],
                    'line_idx': temp_elements[4],
                    'img_width': int(temp_elements[5]),
                    'img_height': int(temp_elements[6]),
                    'box': box_str,
                    'content': original_text
                }
                save_script_to_db(manga_name, manga_id, chapter_idx, [db_entry])

        # 批量更新翻译
        update_translation_in_db(manga_id, chapter_idx, trans_map)
        print(f"已同步更新译文脚本及数据库: {script_zh_path}")

    except Exception as e:
        print(f"写入译文脚本或数据库失败: {e}")

    return final_list


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
    translate_only_chapters = set()  # 新增：专门存放只需补翻译的章节

    # --- 阶段一：流式下载 (修复版逻辑) ---
    while pages_left > 0:
        # 安全熔断：防止无限向后查找章节 (例如找了 10 话都是空的)
        if current_chap > int(start_chapter) + 10:
            print("已连续检测 10 个章节无内容，停止预读。")
            break

        status = download_single_page(base_url, auth_user, auth_pass, manga_name, manga_id, current_chap, current_page)

        if status == 0:
            # 下载成功或图片已存在
            affected_chapters.add(current_chap)
            current_page += 1
            pages_left -= 1

        elif status == 1:
            # 404 本话结束，去下一话
            print(f"章节 {current_chap} 结束，进入下一话...")
            current_chap += 1
            current_page = 0

        elif status == 3:
            # 发现 script.txt 已存在
            path = STORAGE_ROOT / str(manga_name) / str(manga_id) / str(current_chap)
            # 判断是否缺失 script_zh.txt
            if not (path / "script_zh.txt").exists():
                print(f"章节 {current_chap} 存在脚本但缺失翻译，加入补翻队列。")
                translate_only_chapters.add(current_chap)
            else:
                print(f"章节 {current_chap} 已完成(含翻译)，跳过。")

            current_chap += 1
            current_page = 0

        else:
            # 其他错误 (status=2)
            print("遇到下载错误，停止预读。")
            break

    # --- 阶段二：批量 OCR 和 剧本转化  ---
    all_to_translate = affected_chapters | translate_only_chapters

    # 1. 先对新下载的章节跑 OCR
    for chap_idx in affected_chapters:
        chap_dir = STORAGE_ROOT / str(manga_name) / str(manga_id) / str(chap_idx)
        run_mokuro_on_dir(chap_dir)
        generate_script_file(chap_dir, manga_name, manga_id, chap_idx)

    # 2. 对所有需要翻译的章节跑 AI 翻译
    if ai_client:
        for chap_idx in all_to_translate:
            chap_dir = STORAGE_ROOT / str(manga_name) / str(manga_id) / str(chap_idx)
            script_file = chap_dir / "script.txt"

            # 双重保险：确保 script.txt 真的存在
            if script_file.exists():
                with open(script_file, 'r', encoding='utf-8') as f:
                    original_lines = f.readlines()

                print(f"--- 正在通过 AI JSON 模式翻译全话: {manga_name} 第 {chap_idx} 话 ---")
                translate_full_chapter_to_json(ai_client, your_model, manga_name, manga_id, chap_idx, original_lines)

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