# ETYMO — web dashboard (Flask + Gunicorn)
# Build from repo root with trained artifacts present: models/, results/, data/

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Manim + headless OpenGL (xvfb): cairo/pango for text, ffmpeg for video, GL for the renderer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    ffmpeg \
    libcairo2-dev \
    libpango1.0-dev \
    xvfb \
    xauth \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY src ./src
COPY web ./web
COPY manim_scenes ./manim_scenes
COPY models ./models
COPY data ./data
COPY results ./results

EXPOSE 5000

# One worker: Word2Vec models are large; multiple workers would duplicate RAM.
CMD ["gunicorn", "--chdir", "/app/web", "-w", "1", "-b", "0.0.0.0:5000", \
     "--timeout", "120", "--access-logfile", "-", "app:app"]
