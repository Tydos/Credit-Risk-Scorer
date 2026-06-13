# FastAPI Inference Service Image
# --------------------------------
# This image runs a FastAPI-based REST API using Uvicorn.
# It is intended for serving machine learning models or
# backend services in a containerized environment.


# Base image
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffer logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# Copy requirements
COPY services/inference/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy code
COPY services/inference/ .
COPY src/ src/

# Expose port
EXPOSE 8000

# Command to run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

