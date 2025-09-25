import time
import MySQLdb
import os

host = os.environ.get("DB_HOST", "db")
user = os.environ.get("DB_USER", "clothuser")
password = os.environ.get("DB_PASSWORD", "root")
db_name = os.environ.get("DB_NAME", "ecommerce")

while True:
    try:
        conn = MySQLdb.connect(
            host=host, user=user, passwd=password, db=db_name, port=3306
        )
        conn.close()
        print("✅ Database is ready!")
        break
    except Exception as e:
        print("⏳ Waiting for database...", str(e))
        time.sleep(3)
