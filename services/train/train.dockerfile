FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY core/ core/
COPY pipeline/ pipeline/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

COPY services/train/ .

ENTRYPOINT ["python", "train.py"]
