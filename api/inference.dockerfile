# FastAPI Inference Service Image
# --------------------------------
# This image runs a FastAPI-based REST API using Uvicorn.
# It is intended for serving machine learning models or
# backend services in a containerized environment.
#
# Application Server:
#   - FastAPI (ASGI framework)
#   - Uvicorn (ASGI server)
#
# Network:
#   - Exposes HTTP API on port 8000


# Base image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy requirements
COPY api/requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose port
EXPOSE 8000

# Command to run FastAPI with Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

