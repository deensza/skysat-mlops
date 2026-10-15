# Prediction API image. Build context is the project root (see docker-compose.yml).
# The model is NOT baked into the image: the API downloads the `champion` from MLflow at start.
FROM python:3.12.15-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# CPU-only PyTorch (the default wheel on Linux x86 also pulls ~2 GB of CUDA libraries)
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY api/requirements.txt api/requirements.txt
RUN pip install -r api/requirements.txt

# Same preprocessing + model code as training (training/train.py), so serving cannot drift from it
COPY skysat/ skysat/
COPY training/__init__.py training/train.py training/
COPY api/ api/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
