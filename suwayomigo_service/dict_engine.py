import sqlite3
import os


class SQLiteDictEngine:
    def __init__(self, db_filename="manga_dict.db"):
        # 1. 获取当前文件 (dict_engine.py) 的绝对目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 2. 拼接数据库的绝对路径，确保无论在哪里启动都能找到它
        self.db_path = os.path.join(current_dir, db_filename)

        if not os.path.exists(self.db_path):
            print(f"⚠️ 数据库不存在: {self.db_path}，请确认字典文件已放入源码文件夹。")

    def lookup(self, base_form):
        """
        根据原型从 SQLite 检索
        """
        try:
            # 每次查询建立连接（或使用线程池），SQLite 很快
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # 限制返回 1 条最匹配的结果
            cursor.execute('SELECT reading, definition FROM dictionary WHERE term = ? LIMIT 1', (base_form,))
            row = cursor.fetchone()
            conn.close()

            if row:
                return {"r": row[0], "d": row[1]}
        except Exception as e:
            print(f"❌ 查词出错: {e}")

        return {"r": "", "d": ""}


# 实例化
dict_engine = SQLiteDictEngine()