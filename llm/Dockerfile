# Complaint-box API image (llm/serve.py). Build context is the project root (the `api` service of docker-compose.yml).
# No model inside: the base model (Qwen2.5-1.5B-Instruct, ~3 GB) is downloaded at the first start into
# HF_HOME=/models (the llm-models volume, kept between restarts); the LoRA adapter comes from MLflow.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models

# CPU-only PyTorch (the plain PyPI wheel on Linux also pulls ~2 GB of CUDA libraries; the server has no GPU)
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
# torch is pinned again so nothing below can replace the CPU build
RUN pip install torch==2.14.1 transformers==5.20.2 peft==0.21.2 mlflow-skinny==3.17.1 fastapi==0.143.1 \
    uvicorn==0.54.0 pydantic==2.14.1 "psycopg[binary]==3.3.6" psycopg-pool==3.3.3 numpy==2.1.3 pandas==2.2.3

# llm/train_lora.py (imported by serve.py) reads simulator/scenarios.py; nothing else of the repo is needed.
# Never data/ or artifacts/ (the review dataset, local adapters).
COPY llm/ llm/
COPY simulator/__init__.py simulator/scenarios.py simulator/

EXPOSE 8000
CMD ["uvicorn", "llm.serve:app", "--host", "0.0.0.0", "--port", "8000"]
