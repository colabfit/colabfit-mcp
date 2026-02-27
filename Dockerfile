# syntax=docker/dockerfile:1.6

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ARG PYG_WHEEL_INDEX=https://data.pyg.org/whl/torch-2.8.0+cu128.html

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates build-essential wget git cmake && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

# Miniforge for kim-api (conda-forge only, no Anaconda ToS issues)
RUN wget -q https://github.com/conda-forge/miniforge/releases/download/25.1.1-2/Miniforge3-Linux-x86_64.sh -O /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p /opt/conda \
    && rm /tmp/mc.sh \
    && /opt/conda/bin/conda install -y kim-api==2.4.1 kimpy==2.1.3 \
    && /opt/conda/bin/conda clean -afy
ENV PATH="/opt/conda/bin:$PATH"

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install nvidia-cuda-runtime-cu12==12.8.90 nvidia-cuda-nvrtc-cu12==12.8.93 \
    nvidia-cuda-cupti-cu12==12.8.90 nvidia-nvtx-cu12==12.8.90 \
    nvidia-nvjitlink-cu12==12.8.93

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install nvidia-cublas-cu12==12.8.4.1 nvidia-cufft-cu12==11.3.3.83 \
    nvidia-curand-cu12==10.3.9.90 nvidia-cusolver-cu12==11.7.3.90 \
    nvidia-cusparse-cu12==12.5.8.93 nvidia-cusparselt-cu12==0.7.1

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install nvidia-cudnn-cu12==9.10.2.21 nvidia-nccl-cu12==2.27.3 \
    nvidia-cufile-cu12==1.13.1.3
RUN --mount=type=cache,target=/root/.cache/pip pip install torch==2.8.0

# PyG graph dependencies (exact wheel URL for torch 2.8.0 + cu128)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f ${PYG_WHEEL_INDEX}
RUN --mount=type=cache,target=/root/.cache/pip pip install torch_geometric==2.6.1

RUN --mount=type=cache,target=/root/.cache/pip pip install pytorch-lightning==2.5.2
RUN --mount=type=cache,target=/root/.cache/pip pip install jsonargparse==4.40.0
RUN --mount=type=cache,target=/root/.cache/pip pip install torch_ema==0.3.0 tensorboardX

RUN git clone --depth 1 https://github.com/openkim/klay.git /opt/klay \
    && pip install /opt/klay
RUN --mount=type=cache,target=/root/.cache/pip pip install git+https://github.com/openkim/kliff.git

COPY pyproject.toml README.md LICENSE /app/
COPY src/ /app/src/
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/pip pip install '.[full]'

# ---

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 && \
    rm -rf /var/lib/apt/lists/*

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} mcpuser && \
    useradd -u ${USER_ID} -g mcpuser -m -s /bin/bash mcpuser

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/conda /opt/conda

ENV PATH="/opt/venv/bin:/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH:-}"

RUN mkdir -p /home/mcpuser/colabfit/models /home/mcpuser/colabfit/datasets \
    /home/mcpuser/colabfit/inference_output && \
    chown -R mcpuser:mcpuser /home/mcpuser && \
    chmod 755 /home/mcpuser

USER mcpuser
WORKDIR /home/mcpuser

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD /opt/venv/bin/python -c "import colabfit_mcp; print('ok')" || exit 1

ENTRYPOINT ["/opt/venv/bin/colabfit-mcp"]
