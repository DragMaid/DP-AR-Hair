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
    psycopg2-binary \
    python-dotenv \
    fastapi

COPY core /app/core
COPY internal /app/internal
COPY crons/cleaner.py /app/cleaner.py

# Run the cleaner
CMD ["python", "/app/cleaner.py"]
