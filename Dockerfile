FROM python:3.12-slim

WORKDIR /app

# Install system deps for asyncpg (libpq)
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Health check — verifies the API is responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/v1/health'); assert r.status_code == 200" || exit 1

# Use shell form so PORT env var gets expanded by the shell
CMD sh -c 'uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}'

