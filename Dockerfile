FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps (psycopg2 needs these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Env vars will be injected at runtime (DB_DSN, SYMBOLS, etc.)
# Example defaults (can be overridden)
ENV QUOTE_INTERVAL_MS=1000
ENV WITH_FUNDING=true

# Main entrypoint
CMD ["python", "listener/binance_listener.py"]
