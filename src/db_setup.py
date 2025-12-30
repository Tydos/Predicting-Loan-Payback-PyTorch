import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Aurora/PostgreSQL connection parameters
HOST = "loan-prediction-metadata.c4jskc8imd2i.us-east-1.rds.amazonaws.com"
USER = "postgresql"            # your DB username
PASSWORD = "Prasadj38!"  # DB password
PORT = 5432                # default PostgreSQL port
DB_NAME = "mlflow"         # the new database to create

# Connect to the default database 'postgres'
conn = psycopg2.connect(
    host=HOST,
    database="postgres",  # must connect to an existing DB first
    user=USER,
    password=PASSWORD,
    port=PORT
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
