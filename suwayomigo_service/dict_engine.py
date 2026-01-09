import sqlite3
import os


class SQLiteDictEngine:
    def __init__(self, db_path="manga_dict.db"):
        self.db_path = db_path
        if not os.path.exists(db_path):
            print(f"⚠️ 数据库不存在: {db_path}，请先运行转换脚本")

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