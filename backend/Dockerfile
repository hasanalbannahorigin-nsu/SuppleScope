# SuppleScope back end (FastAPI). Build from the repository root:  docker compose build backend
FROM python:3.10-slim

ARG INSTALL_ML=false
ARG TESSDATA_BEST=true
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-ben tesseract-ocr-eng ffmpeg libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

# Tesseract "best" (LSTM) models for Bangla and English, recommended in the stack report.
RUN if [ "$TESSDATA_BEST" = "true" ]; then \
        mkdir -p /opt/tessdata_best && \
        curl -fsSL -o /opt/tessdata_best/ben.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/ben.traineddata && \
        curl -fsSL -o /opt/tessdata_best/eng.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata && \
        curl -fsSL -o /opt/tessdata_best/osd.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/osd.traineddata ; \
    fi
ENV TESSDATA_PREFIX=/opt/tessdata_best

WORKDIR /app/backend
COPY backend/requirements.txt backend/requirements-ml.txt ./
RUN pip install -r requirements.txt \
    && if [ "$INSTALL_ML" = "true" ]; then \
         pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-ml.txt ; \
       fi

COPY backend/ ./
RUN mkdir -p data/storage data/cache data/knowledge models

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -fs http://localhost:8000/api/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
