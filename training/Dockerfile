# CT worker image (step 8): training/ct_worker.py, also used for one-off runs (prepare_data, retrain).
# Build context is the project root (docker-compose.yml).
# The Kaggle files are NOT in the image (.dockerignore excludes data/): docker-compose.yml mounts data/.
FROM python:3.12.15-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/tmp

# CPU-only PyTorch, like the API image (see api/Dockerfile)
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install torch==2.14.1 --index-url ${TORCH_INDEX}
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt "psycopg[binary]==3.3.6"

COPY skysat/ skysat/
COPY training/ training/

CMD ["python", "-m", "training.ct_worker"]
