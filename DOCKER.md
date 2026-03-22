# Docker

The image runs the **Flask dashboard** with **Gunicorn**. It includes **Manim**, **ffmpeg**, **xvfb** (virtual framebuffer), and **`manim_scenes/`** so on-demand renders (`/api/render/...`) work in headless Linux. Pre-rendered MP4s under `results/videos/` are still used when present.

## Prerequisites

Before building, your repo should include the artifacts the app loads at startup:

- `models/embeddings/` — `w2v_old.model`, `w2v_new.model`, `w2v_old_aligned.npy`
- `models/etymology_classifier.pkl` (optional but recommended)
- `data/` — e.g. `selected_words.csv` and other CSVs referenced in `src/config.py`
- `results/` — `drift_scores.csv`, `origin_drift_summary.csv`, `umap_coords.csv`, videos under `results/videos/`, etc.

Generate these with your normal pipeline (`build_dataset`, `train_embeddings`, `compute_drift`, `classifier`, …) if they are not already present.

## Build and run

```bash
docker build -t etymo .
docker run --rm -p 5000:5000 etymo
```

Open `http://localhost:5000`.

## Docker Compose

```bash
docker compose up --build
```

## Compose with bind mounts

If binaries are large and you prefer not to copy them into the image, uncomment the `volumes` section in `docker-compose.yml` and add this to `.dockerignore` so the context stays small:

```
models
results
data
```

Then build still copies `src/` and `web/`; at runtime the host folders are mounted.

## Notes

- **Memory:** Word2Vec + aligned vectors need substantial RAM; allocate enough for Docker (often 4GB+).
- **Workers:** The Dockerfile uses **one** Gunicorn worker to avoid loading duplicate copies of the embedding models.
