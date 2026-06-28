# Training server
# Runs: python -m src.main
# Expects volumes from docker-compose.yml:
#   ./dataset:/app/dataset:ro

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY services/train/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY services/train/ .
COPY src/ src/

ENTRYPOINT ["python", "train.py"]