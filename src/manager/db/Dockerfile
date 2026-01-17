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
    pillow=="10.3.0" \
    httpx=="0.28.1" \
    pyjwt=="2.10.1" \
    cachetools=="6.2.4" \
    starlette=="0.50.0" \
    uvicorn=="0.40.0" \
    python-multipart=="0.0.21" \
    pydantic=="2.7.1" \
    pydantic-core=="2.18.2"


# Run the server
CMD ["uvicorn", "manager.server:app", "--host", "0.0.0.0", "--port", "8000"]
