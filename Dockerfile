# ── ResearchMind API image ────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Optional offline/free embedding fallback. Disabled by default to keep the
# image small (sentence-transformers pulls in PyTorch).
ARG INSTALL_LOCAL_EMBEDDINGS=false
RUN if [ "$INSTALL_LOCAL_EMBEDDINGS" = "true" ]; then \
        pip install langchain-huggingface>=1.2.0 sentence-transformers>=3.0.0; \
    fi

COPY . .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
