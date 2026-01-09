FROM postgres:16.9-bookworm

# Install build dependencies and pg_cron extension
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-server-dev-16 \
        postgresql-16-cron \
        build-essential \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*
