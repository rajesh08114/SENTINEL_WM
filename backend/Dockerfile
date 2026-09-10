# SENTINEL-WM inference backend - fully self-contained (no research tree).
#   docker build -t sentinel-wm-backend backend/
# `backend/models/` (the bundle, if present) is COPIED in, so the image runs
# standalone. To update models without a rebuild, mount over it:
#   docker run -p 8000:8000 -v "$PWD/backend/models:/app/models:ro" sentinel-wm-backend
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SENTINEL_WM_MODEL_DIR=/app/models \
    SENTINEL_JOB_DIR=/app/data/jobs \
    SENTINEL_DEVICE=cpu

WORKDIR /app

# CPU-only torch wheel first (keeps the image ~1 GB smaller than the CUDA build)
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1"

COPY . /app

RUN pip install . \
 && useradd -m -u 10001 sentinel \
 && mkdir -p /app/data/jobs /app/models \
 && chown -R sentinel:sentinel /app

USER sentinel
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
