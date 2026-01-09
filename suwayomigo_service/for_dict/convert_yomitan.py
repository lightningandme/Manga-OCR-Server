import zipfile
import json
import sqlite3
import os


def create_database(db_name="manga_dict.db"):
    # 如果数据库已存在，先删除以确保结构更新（或者你可以手动删除）
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 核心修正：创建 4 列的表结构
    cursor.execute('DROP TABLE IF EXISTS dictionary')
    cursor.execute('''
                   CREATE TABLE dictionary
                   (
                       term       TEXT,
                       reading    TEXT,
                       pos        TEXT,
                       definition TEXT
                   )
                   ''')
    # 为 term 创建索引 (Index) 以提升查询速度
    cursor.execute('CREATE INDEX idx_term ON dictionary(term)')
    conn.commit()
    return conn


def process_zip(zip_path, conn):
    cursor = conn.cursor()
    print(f"📦 正在转换: {zip_path}")

    with zipfile.ZipFile(zip_path, 'r') as z:
        bank_files = [f for f in z.namelist() if f.startswith('term_bank')]
        for bank_file in bank_files:
            with z.open(bank_file) as f:
                data = json.load(f)
                insert_data = []
                for entry in data:
                    term = entry[0]
                    reading = entry[1]
                    pos = entry[2]  # 词性标签 (POS Tag)

                    # 递归解析逻辑，防止结构化 JSON 导致的显示乱码
                    def parse_definition(d):
                        if isinstance(d, list):
                            return "\n".join([parse_definition(i) for i in d])
                        if isinstance(d, dict):
                            # 尝试提取 Yomitan 富文本中的内容
                            return str(d.get('content', d.get('text', d)))
                        return str(d)

                    definition = parse_definition(entry[5])
                    # 确保这里是 4 个元素，对应数据库的 4 列
                    insert_data.append((term, reading, pos, definition))

                # 批量插入 (Bulk Insert)
                cursor.executemany('INSERT INTO dictionary VALUES (?, ?, ?, ?)', insert_data)
    conn.commit()


if __name__ == "__main__":
    db_conn = create_database()
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    found_zip = False
    for file in os.listdir(current_dir):
        if file.endswith(".zip"):
            found_zip = True
            process_zip(os.path.join(current_dir, file), db_conn)

    if not found_zip:
        print("❌ 未在当前目录找到 .zip 词典文件")

    db_conn.close()
    print("✨ 转换完成！已生成包含词性(POS)的 manga_dict.db")