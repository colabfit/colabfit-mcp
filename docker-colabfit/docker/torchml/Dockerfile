# starting from KDP-torchml v1.8.9
FROM ghcr.io/openkim/developer-platform:v1.8.9-torchml

# temporarily switch to ROOT user to install systemwide colabfit-mcp requirements
USER root

# colabfit sys installations
RUN apt-get update && apt-get install -y \
    libgomp1 \
    nano \
    && rm -rf /var/lib/apt/lists/*


ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# this upgrades pip-26 much like kdp
# kdp must be doing this at a later stage probably because kdp-sys is only at pip-22
RUN python -m pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel \
    build

# use KDP packages directory
ARG PACKAGE_DIR=/pipeline/packages/

# colabfit-mcp repo url
ARG REPO_URL=https://github.com/colabfit/colabfit-mcp.git
# ARG REPO_REF=main
ARG REPO_REF=kdp_minimal_base

# Build Torch libraries for common Nvidia configs by default
ENV TORCH_CUDA_ARCH_LIST=Common

# Torch 2.8.0 — KDP base ships no torch; install explicitly to match root Dockerfile.
# GPU (default): TORCH_VARIANT=cu128. CPU: --build-arg TORCH_VARIANT=cpu
# with --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VARIANT=cu128
# ARG TORCH_VARIANT=cpu
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_VARIANT}"
RUN if [ -n "${TORCH_INDEX_URL}" ]; then \
        python -m pip install --no-cache-dir torch==2.8.0 --index-url "${TORCH_INDEX_URL}"; \
    else \
        python -m pip install --no-cache-dir torch==2.8.0; \
    fi
RUN python -m pip install --no-cache-dir \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f "https://data.pyg.org/whl/torch-2.8.0+${TORCH_VARIANT}.html"

# install dependencies pinned to a version
RUN python -m pip install "mcp[cli]>=1.0.0,<2"
RUN python -m pip install "requests==2.33.1"
RUN python -m pip install "numpy==2.0.0"
RUN python -m pip install "pyarrow==24.0.0"
RUN python -m pip install "ase==3.25.0"
RUN python -m pip install "kliff @ git+https://github.com/gpwolfe/kliff.git@colabfit_mcp_edit"
RUN python -m pip install "datasets>=2.0"
RUN python -m pip install "klay @ git+https://github.com/gpwolfe/klay.git@colabfit_mcp_edits"
RUN python -m pip install "pytorch-lightning>=2.5.2" "jsonargparse==4.40.0"
RUN python -m pip install "torch_geometric==2.6.1"
RUN python -m pip install "torch_ema==0.3" "tensorboardX"
RUN python -m pip install "kimpy==2.1.4"
RUN python -m pip install "kim-tools==0.4.9"
RUN python -m pip install "kimvv==0.1.10"

# Source code comes last — all layers above are cached when only src changes.
# Clone repo into container
# RUN git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} ${PACKAGE_DIR}/colabfit-mcp

# For dev only
COPY . ${PACKAGE_DIR}/colabfit-mcp
WORKDIR ${PACKAGE_DIR}/colabfit-mcp

# Install colabfit-mcp from local source; all deps already installed above
RUN pip install --no-deps .

RUN chown -R openkim /pipeline/packages/colabfit-mcp

USER openkim
WORKDIR /home/openkim

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import colabfit_mcp; import torch; print(torch.__version__)" || exit 1

ENTRYPOINT ["colabfit-mcp"]

#############################################
