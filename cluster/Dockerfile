FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    CCACHE_DIR=/root/.ccache \
    TORCH_CUDA_ARCH_LIST="7.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python-is-python3 \
        python3-venv \
        build-essential \
        clang \
        ccache \
        curl \
        git \
        ca-certificates \
        unzip \
        libomp-dev \
        libgl1 \
        libglvnd0 \
        libx11-6 \
        libxrandr2 \
        libxi6 \
        libxcursor1 \
        libxinerama1 \
        libxxf86vm1 \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

WORKDIR /workspace/PufferLib
COPY . /workspace/PufferLib

# Install a CUDA runtime compatible with the cluster's 535 driver
# (nvidia-smi reports CUDA 12.2). Do not let the editable install upgrade torch.
RUN uv pip install \
        --index-strategy unsafe-best-match \
        --index-url https://download.pytorch.org/whl/cu121 \
        --extra-index-url https://pypi.org/simple \
        torch==2.5.1+cu121 \
        torchvision==0.20.1+cu121 \
        torchaudio==2.5.1+cu121

RUN uv pip install \
        pip \
        "setuptools>=69" \
        wheel \
        numpy \
        rich \
        rich_argparse \
        gpytorch \
        scikit-learn \
        wandb \
        pybind11

RUN uv pip install --no-deps -e . --no-build-isolation

RUN OMP_LIBDIR="$(dirname "$(find /usr -name libomp.so -print -quit)")" \
    && EXTRA_LDFLAGS="-L${OMP_LIBDIR}" NVCC_ARCH=sm_70 bash build.sh quad_meshing --float

RUN mkdir -p /outputs

ENV PUFFERLIB_ROOT=/workspace/PufferLib

CMD ["puffer", "train", "quad_meshing", "--slowly"]
