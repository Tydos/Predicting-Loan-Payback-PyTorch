import psycopg2
import dotenv
import os
dotenv.load_dotenv()
conn = psycopg2.connect(
    host=os.getenv("HOST"),
    database=os.getenv("DATABASE"),
    user=os.getenv("USER"),
    password=os.getenv("PASSWORD")
)
cur = conn.cursor()
cur.execute("SELECT * FROM runs LIMIT 10;")  # Example table
rows = cur.fetchall()

for row in rows:
    print(row)

cur.close()
conn.close()
