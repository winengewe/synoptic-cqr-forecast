FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install OpenMP runtime required by XGBoost and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 1. Cache pip dependencies layer
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy internal package source and install in editable mode
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .

# 3. Pre-create artifact directories so Docker doesn't mount them with root-only lockouts
RUN mkdir -p /app/data /app/models

# 4. Copy top-level entry point scripts
COPY main.py predict_tomorrow.py ./

CMD ["python", "predict_tomorrow.py"]