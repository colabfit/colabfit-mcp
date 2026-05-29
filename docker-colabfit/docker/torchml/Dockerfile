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
RUN python -m pip install --user --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel \
    build

# use KDP packages directory
ARG PACKAGE_DIR=/pipeline/packages/ 

# colabfit-mcp repo url
ARG REPO_URL=https://github.com/colabfit/colabfit-mcp.git
ARG REPO_REF=kdp_minimal_base

# Clone repo into container
RUN git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} ${PACKAGE_DIR}/colabfit-mcp

# THIS IS PURELY FOR DEBUGGING!!!! DONT COMMIT THIS!!!
# COPY . /pipeline/packages/colabfit-mcp

# set working directory to /pipeline/packages
WORKDIR ${PACKAGE_DIR}/colabfit-mcp


# install openai-agent library. This probably doesnt belong here. This is client-side code. Not server code. 
# RUN python -m pip install --no-cache-dir openai-agents

# Build Torch libraries for common Nvidia configs by default
ENV TORCH_CUDA_ARCH_LIST=Common
# CPU torch v2.8.0
RUN pip install --no-cache-dir torch==2.8.0
# pyg_lib 0.6.0+pt28cpu
RUN pip install --no-cache-dir pyg_lib -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
# torch_scatter 2.1.2+pt28cpu
RUN pip install --no-cache-dir torch_scatter -f https://data.pyg.org/whl/torch-2.8.0+cpu.html 
# torch_sparse 0.6.18+pt28cpu
RUN pip install --no-cache-dir torch_sparse -f https://data.pyg.org/whl/torch-2.8.0+cpu.html 
# torch_cluster 1.6.3+pt28cpu
RUN pip install --no-cache-dir torch_cluster -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
# torch_spline_conv 1.2.2+pt28cpu
RUN pip install --no-cache-dir torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

# install dependencies pinned to a version
RUN python -m pip install --no-cache-dir "mcp[cli]>=1.0.0"
RUN python -m pip install --no-cache-dir "requests==2.33.1"
RUN python -m pip install --no-cache-dir "numpy==2.0.0"
RUN python -m pip install --no-cache-dir "pyarrow==24.0.0"
RUN python -m pip install --no-cache-dir "ase==3.25.0"
RUN python -m pip install --no-cache-dir "kliff @ git+https://github.com/gpwolfe/kliff.git@colabfit_mcp_edit"
RUN python -m pip install --no-cache-dir "datasets>=2.0"
RUN python -m pip install --no-cache-dir "klay @ git+https://github.com/gpwolfe/klay.git@colabfit_mcp_edits"
RUN python -m pip install --no-cache-dir "torch_geometric==2.6.1"
RUN python -m pip install --no-cache-dir "pytorch-lightning>=2.5.2"
RUN python -m pip install --no-cache-dir "torch_ema==0.3"
RUN python -m pip install --no-cache-dir "kimpy==2.1.4"
RUN python -m pip install --no-cache-dir "kim-tools==0.4.9"
RUN python -m pip install --no-cache-dir "kimvv==0.1.10"

# Install colabfit-mcp and dependencies from repo
# RUN pip install --user --no-cache-dir "colabfit_mcp[full] @ git+${REPO_URL}@${REPO_REF}"

# finally install source code and any further required dependancies
# we install only the [minimal] version although this is [torchml] version container
# dont install [full] because [full] is not supposed to be pinned to a specific version (i think)
RUN pip install .


HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import colabfit_mcp; import torch; print(torch.__version__)" || exit 1

ENTRYPOINT ["colabfit-mcp"]

#############################################
