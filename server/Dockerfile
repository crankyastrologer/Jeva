FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY requirements.txt ./requirements-repo.txt
RUN pip install --upgrade pip && pip install -r requirements-repo.txt

COPY server ./server
RUN pip install ./server

ENV JEVA_MODEL=/models/Jeva-4B \
    JEVA_HOST=0.0.0.0 \
    JEVA_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

CMD ["uvicorn", "jevad.app:app", "--app-dir", "/app/server", "--host", "0.0.0.0", "--port", "8000"]
