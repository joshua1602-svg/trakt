# Trakt MI Agent API — container image (Azure App Service for Containers).
# Image name: trakt-mi-api
#
# Build context is the REPO ROOT (so mi_agent_api, mi_agent, apps/blob_trigger_app,
# engine and config all ship). Build with:
#   az acr build --registry traktregistry1602 --image trakt-mi-api:v1 \
#       --file deploy/trakt-mi-api/Dockerfile .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    MI_AGENT_SCRATCH=/tmp/trakt/mi_platform

WORKDIR /app

# Build deps for any source-only wheels (lxml/scientific stack); removed after.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libxml2-dev libxslt1-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching. The root requirements.txt carries
# the full runtime set PLUS the API server (fastapi / uvicorn[standard] /
# gunicorn / uvicorn-worker).
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# Application code (whole repo; see .dockerignore for exclusions).
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/health" || exit 1

# Start command (exactly as required; $PORT is provided by App Service).
CMD ["sh", "-c", "gunicorn --bind=0.0.0.0:$PORT -k uvicorn.workers.UvicornWorker mi_agent_api.app:app --workers ${MI_API_WORKERS:-2} --timeout ${MI_API_TIMEOUT:-120}"]
