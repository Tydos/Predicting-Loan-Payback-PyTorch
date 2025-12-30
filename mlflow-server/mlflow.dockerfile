FROM python:3.11-slim

WORKDIR /app

COPY mlflow-server/requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


EXPOSE 5000

# Start MLflow server on EC2
# CMD ["mlflow", "server","--backend-store-uri", "${MLFLOW_BACKEND_STORE_URI}","--default-artifact-root", "${ARTIFACT_ROOT}","--host", "0.0.0.0", "--port", "5000","--allowed-hosts", "*"]
CMD mlflow server \
    --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
    --default-artifact-root $ARTIFACT_ROOT \
    --host 0.0.0.0 \
    --port 5000 \
    --allowed-hosts '*'