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
ARG REPO_REF=kdp_minimal_base

# use KDP packages directory
ARG PACKAGE_DIR=/pipeline/packages/ 


# colabfit-mcp repo url
ARG REPO_URL=https://github.com/colabfit/colabfit-mcp.git
ARG REPO_REF=kdp_minimal_base

# Clone repo into container
RUN git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} ${PACKAGE_DIR}/colabfit-mcp

# set working directory to /pipeline/packages
WORKDIR ${PACKAGE_DIR}/colabfit-mcp


# install openai-agent library. This probably doesnt belong here. This is client-side code. Not server code. 
# RUN python -m pip install --no-cache-dir openai-agents

# CPU torch
# RUN python -m pip install --user --no-cache-dir \
#     torch --index-url https://download.pytorch.org/whl/cpu

# PyG wheels matched to torch 2.8.0 CPU
# NOTE: THESE SHOULD HAVE VERSIONS AFFIXED TO THEM BECAUSE THEIR DOCKER-IMAGE INSTALLATIONS
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
    torch_ema==0.3.0 
    # klay # dont install klay here. install gregory's fork of klay

# install dependencies pinned to a version
RUN python -m pip install --user --no-cache-dir "mcp[cli]>=1.0.0"
RUN python -m pip install --user --no-cache-dir "requests==2.33.1"
RUN python -m pip install --user --no-cache-dir "numpy==2.0.0"
RUN python -m pip install --user --no-cache-dir "pyarrow==24.0.0"
RUN python -m pip install --user --no-cache-dir "ase==3.25.0"
RUN python -m pip install --user --no-cache-dir "kliff @ git+https://github.com/gpwolfe/kliff.git@colabfit_mcp_edit"
RUN python -m pip install --user --no-cache-dir "datasets>=2.0"
RUN python -m pip install --user --no-cache-dir "klay @ git+https://github.com/gpwolfe/klay.git@colabfit_mcp_edits"
RUN python -m pip install --user --no-cache-dir "pytorch-lightning>=2.0"
RUN python -m pip install --user --no-cache-dir "torch_geometric==2.6.1"
RUN python -m pip install --user --no-cache-dir "torch_scatter"
RUN python -m pip install --user --no-cache-dir "torch_ema==0.3"
RUN python -m pip install --user --no-cache-dir "kimpy==2.1.4"
RUN python -m pip install --user --no-cache-dir "kim-tools==0.4.9"
RUN python -m pip install --user --no-cache-dir "kimvv==0.1.10"

# Install colabfit-mcp and dependencies from repo
# RUN pip install --user --no-cache-dir "colabfit_mcp[full] @ git+${REPO_URL}@${REPO_REF}"

# finally install source code and any further required dependancies
# we install only the [minimal] version although this is [torchml] version container
# dont install [full] because [full] is not supposed to be pinned to a specific version (i think)
RUN pip install --user --no-cache-dir .

# set working directory to /pipeline/packages
WORKDIR ${PACKAGE_DIR}/colabfit-mcp


WORKDIR /home/openkim

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import colabfit_mcp; import torch; print(torch.__version__)" || exit 1

ENTRYPOINT ["/home/openkim/.local/bin/colabfit-mcp"]

#############################################
