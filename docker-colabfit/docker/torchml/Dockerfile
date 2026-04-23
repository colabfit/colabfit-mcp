# starting from KDP-torchml
FROM ghcr.io/openkim/developer-platform:latest-torchml

# temporarily switch to ROOT user to install systemwide colabfit-mcp requirements
USER root 

# colabfit sys installations
RUN apt-get update && apt-get install -y \
    libgomp1 \
    nano \
    && rm -rf /var/lib/apt/lists/*

# revert back to openkim user
USER openkim

# add installation path of colabfit-mcp executable to $PATH
ENV PATH="/home/openkim/.local/bin:${PATH}"
    # kliff and datasets library are installed in /home/openkim/.local/lib/python3.10/site-packages 
    # whereas the KDP's pip libraries (kimpy, kim_tools) are in /usr/local/lib/python3.10/dist-packages/
        # how did KDP pip install in the root directory? 
    # do we want kliff and datasets in the same directory as the other pip libraries? 
        # how do we safely achieve that?
    # this is not necessarily a bad thing. 
        # it helps keep separate KDP's pip libraries and colabfit's pip libraries. 
        # as long as python3 is able to find both directories, which it can as of now

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# this upgrades pip-26 much like kdp
# kdp must be doing this at a later stage probably because kdp-sys is only at pip-22
RUN python -m pip install --user --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel \
    build



# colabfit-mcp repo url
ARG REPO_URL=https://github.com/colabfit/colabfit-mcp.git
ARG REPO_REF=main

# use KDP packages directory
ARG PACKAGE_DIR=/pipeline/packages/ 

# Clone repo into container
RUN git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} ${PACKAGE_DIR}/colabfit-mcp

# set working directory to /pipeline/packages
WORKDIR ${PACKAGE_DIR}/colabfit-mcp


# install openai-agent library. This probably doesnt belong here. This is client-side code. Not server code. 
RUN python -m pip install --no-cache-dir openai-agents

# CPU torch
RUN python -m pip install --user --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# PyG wheels matched to torch 2.8.0 CPU
RUN python -m pip install --user --no-cache-dir \
    pyg_lib \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

RUN python -m pip install --user --no-cache-dir \
    torch_geometric==2.6.1 \
    pytorch-lightning==2.5.2 \
    torch_ema==0.3.0 \
    klay


# install [basic] packages listed in ${PACKAGE_DIR}/colabfit-mcp/pyproject.toml
RUN python -m pip install --no-cache-dir "mcp[cli]>=1.0.0"
RUN python -m pip install --no-cache-dir "requests"
RUN python -m pip install --no-cache-dir "numpy"
RUN python -m pip install --no-cache-dir "pyarrow"
RUN python -m pip install --no-cache-dir "ase"


# install [full] packages listed in ${PACKAGE_DIR}/colabfit-mcp/pyproject.toml
RUN python -m pip install --no-cache-dir "klay @ git+https://github.com/gpwolfe/klay.git@colabfit_mcp_edits"
RUN python -m pip install --no-cache-dir "kliff @ git+https://github.com/gpwolfe/kliff.git@colabfit_mcp_edit"
RUN python -m pip install --no-cache-dir "pytorch-lightning>=2.0"
RUN python -m pip install --no-cache-dir "torch_geometric"
RUN python -m pip install --no-cache-dir "torch_scatter"
RUN python -m pip install --no-cache-dir "torch_ema"
RUN python -m pip install --no-cache-dir "kimpy"
RUN python -m pip install --no-cache-dir "ase"
RUN python -m pip install --no-cache-dir "datasets>=2.0"
RUN python -m pip install --no-cache-dir "huggingface_hub"
RUN python -m pip install --no-cache-dir "kim-tools"
RUN python -m pip install --no-cache-dir "kimvv"


WORKDIR /home/openkim

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import colabfit_mcp; import torch; print(torch.__version__)" || exit 1

ENTRYPOINT ["/home/openkim/.local/bin/colabfit-mcp"]

#############################################
