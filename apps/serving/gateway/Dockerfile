FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1

RUN adduser --disabled-password --gecos "" --uid 10001 app

WORKDIR /app

COPY --chown=app:app apps/serving/gateway/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY --chown=app:app src/shared/ ./src/shared
COPY --chown=app:app apps/serving/gateway/ ./apps/serving/gateway/

USER app

CMD ["python", "-m", "apps.serving.gateway.app"]