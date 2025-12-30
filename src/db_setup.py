import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import dotenv
import os

DB_NAME = "mlflow"         # the new database to create
dotenv.load_dotenv()
conn = psycopg2.connect(
    host=os.getenv("HOST"),
    database=os.getenv("DATABASE"),
    user=os.getenv("USER"),
    password=os.getenv("PASSWORD")
)

# Set autocommit to True to allow CREATE DATABASE
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

# Create a cursor object
cur = conn.cursor()

# Create the new database
cur.execute(f"CREATE DATABASE {DB_NAME};")
print(f"Database '{DB_NAME}' created successfully.")

# Close connections
cur.close()
conn.close()
