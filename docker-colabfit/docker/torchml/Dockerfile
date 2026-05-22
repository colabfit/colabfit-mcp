# starting from KDP-torchml
FROM ghcr.io/openkim/developer-platform:latest-torchml

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
ARG REPO_REF=kdp_minimal_base_claude

# Clone repo into container
RUN git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} ${PACKAGE_DIR}/colabfit-mcp

# For dev only
# COPY . ${PACKAGE_DIR}/colabfit-mcp

# set working directory to /pipeline/packages
WORKDIR ${PACKAGE_DIR}/colabfit-mcp


# install openai-agent library. This probably doesnt belong here. This is client-side code. Not server code. 
# RUN python -m pip install --no-cache-dir openai-agents

# Torch 2.8.0 — KDP base ships no torch; install explicitly to match root Dockerfile.
# GPU (default): TORCH_VARIANT=cu128. CPU: --build-arg TORCH_VARIANT=cpu
# with --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VARIANT=cu128
ARG TORCH_INDEX_URL=
RUN if [ -n "${TORCH_INDEX_URL}" ]; then \
        python -m pip install --no-cache-dir torch==2.8.0 --index-url "${TORCH_INDEX_URL}"; \
    else \
        python -m pip install --no-cache-dir torch==2.8.0; \
    fi
RUN python -m pip install --no-cache-dir \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f "https://data.pyg.org/whl/torch-2.8.0+${TORCH_VARIANT}.html"

# RUN python -m pip install -vvv --user --no-cache-dir \
    # torch_geometric==2.6.1 \
    # pytorch-lightning==2.5.2 \
    # torch_ema==0.3.0 
    # klay # dont install klay here. install gregory's fork of klay

# install dependencies pinned to a version
RUN python -m pip install "mcp[cli]>=1.0.0"
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

# Install colabfit-mcp and dependencies from repo
# RUN pip install --user --no-cache-dir "colabfit_mcp[full] @ git+${REPO_URL}@${REPO_REF}"

# finally install source code and any further required dependancies
# we install only the [minimal] version although this is [torchml] version container
# dont install [full] because [full] is not supposed to be pinned to a specific version (i think)
RUN pip install .

RUN chown -R openkim /pipeline/packages/colabfit-mcp

USER openkim
WORKDIR /home/openkim

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import colabfit_mcp; import torch; print(torch.__version__)" || exit 1

ENTRYPOINT ["colabfit-mcp"]

#############################################
