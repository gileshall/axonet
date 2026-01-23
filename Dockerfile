# Multi-stage Dockerfile for axonet
# Supports: dataset generation, training, and inference

# ==============================================================================
# Base image with system dependencies
# ==============================================================================
FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libegl1 \
    libegl-mesa0 \
    libgles2 \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ==============================================================================
# Dependencies layer (cached separately for faster rebuilds)
# ==============================================================================
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Additional dependencies for cloud and CLIP
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    sentence-transformers \
    pyyaml \
    google-cloud-storage \
    google-cloud-batch \
    google-cloud-compute \
    litellm \
    wandb \
    moderngl

# ==============================================================================
# Development image (includes dev tools)
# ==============================================================================
FROM deps AS dev

RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    ipython \
    jupyter

COPY . .
RUN pip install -e .

CMD ["bash"]

# ==============================================================================
# Dataset generation image (optimized for batch processing)
# ==============================================================================
FROM deps AS dataset

COPY axonet/ ./axonet/
COPY setup.py requirements.txt README.md ./
RUN pip install -e .

ENV AXONET_MODE=dataset
ENTRYPOINT ["python", "-m", "axonet.cloud.entrypoints.generate_dataset"]

# ==============================================================================
# Training image (with GPU support when run on GPU instances)
# ==============================================================================
FROM deps AS train

COPY axonet/ ./axonet/
COPY configs/ ./configs/
COPY setup.py requirements.txt README.md ./
RUN pip install -e .

ENV AXONET_MODE=train
ENTRYPOINT ["python", "-m", "axonet.cloud.entrypoints.train"]

# ==============================================================================
# Production/inference image (minimal)
# ==============================================================================
FROM deps AS prod

COPY axonet/ ./axonet/
COPY setup.py requirements.txt README.md ./
RUN pip install -e .

CMD ["python", "-m", "axonet"]
