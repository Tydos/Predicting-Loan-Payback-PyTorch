import psycopg2

conn = psycopg2.connect(
    host="loan-prediction-metadata.c4jskc8imd2i.us-east-1.rds.amazonaws.com",
    database="mlflow",
    user="postgresql",
    password="Prasadj38!"
)
print("Connected!")
conn.close()
