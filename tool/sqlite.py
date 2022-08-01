from pathlib import Path, PurePosixPath
import sqlite3

db_file = PurePosixPath(Path(__file__).parent.parent, 'resources/db/flower_info.db')


# 数据库插入数据
def insert(sql, data=None):
    connect = sqlite3.connect(db_file)
    cursor = connect.cursor()
    # 判断data是不是列表，是列表就调用executemany方法，不是列表就调用execute方法
    if isinstance(data, list):
        cursor.executemany(sql, data)
    elif data:
        cursor.execute(sql, data)
    else:
        cursor.execute(sql)
    connect.commit()
    # if cursor.rowcount > 0:
    #     print(f'成功在数据库中插入{cursor.rowcount}行数据！')
    cursor.close()
    connect.close()


# 数据库删除数据
def delete(sql, data=None):
    connect = sqlite3.connect(db_file)
    cursor = connect.cursor()
    if data:
        cursor.execute(sql, data)
    else:
        cursor.execute(sql)
    connect.commit()
    # if cursor.rowcount > 0:
    #     print(f'成功在数据库中删除{cursor.rowcount}行数据！')
    cursor.close()
    connect.close()


# 查询数据库
def check(sql, data=None):
    connect = sqlite3.connect(db_file)
    cursor = connect.cursor()
    if data:
        cursor.execute(sql, data)
    else:
        cursor.execute(sql)
    info = cursor.fetchall()
    cursor.close()
    connect.close()
    return info
