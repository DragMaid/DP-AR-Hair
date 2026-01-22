FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies required by psycopg2
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq5 \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    psycopg2-binary=="2.9.11" \
    dotenv=="0.9.9" \
    fastapi=="0.128.0" \
    schedule=="1.2.2"

COPY manager/core /app/manager/core
COPY manager/internal /app/manager/internal
COPY manager/crons/cleaner.py /app/cleaner.py

WORKDIR /app

# Run the cleaner
CMD ["python", "cleaner.py"]
