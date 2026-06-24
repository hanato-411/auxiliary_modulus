FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    PIP_NO_CACHE_DIR=off PYTHONDONTWRITEBYTECODE=1

# 1) OS + SageMath
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates wget gnupg software-properties-common build-essential && \
    add-apt-repository -y universe && \
    apt-get update && \
    apt-get install -y --no-install-recommends sagemath && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g $GROUP_ID appuser && \
    useradd -u $USER_ID -g $GROUP_ID -m -s /bin/bash appuser

WORKDIR /app
RUN chown -R appuser:appuser /app

# 2) pip + base dependencies
RUN sage -pip install --upgrade pip
RUN sage -pip install \
        --ignore-installed \
        --break-system-packages \
        "sympy>=1.13.3"
RUN sage -pip install \
        --break-system-packages \
        "numpy>=1.17.3,<1.25.0" \
        "scipy>=1.7.0,<1.12.0" \
        "transformers>=4.49.0" \
        "omegaconf>=2.3.0" \
        "wandb>=0.15.11" \
        "accelerate>=0.29.0" \
        "joblib>=1.5.0" \
        "scikit-learn>=1.0.0" \
        "tqdm>=4.64.0"
RUN sage -pip install --break-system-packages "pillow>=9.1.0"

# 3) torch (GPU/CPU switch)
ARG TORCH_VARIANT=gpu
ARG TORCH_VERSION=2.6.0
ARG TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu124
RUN if [ "$TORCH_VARIANT" = "gpu" ]; then \
        sage -pip install --break-system-packages --extra-index-url $TORCH_CUDA_INDEX "torch==${TORCH_VERSION}"; \
    else \
        sage -pip install --break-system-packages "torch==${TORCH_VERSION}"; \
    fi

# 4) compatibility packages used by training stack
RUN sage -pip uninstall -y send2trash || true
RUN sage -pip install --break-system-packages "send2trash>=1.8.0"
RUN sage -pip install --break-system-packages "calt-x==1.1.0"

USER appuser
CMD ["/bin/bash"]
