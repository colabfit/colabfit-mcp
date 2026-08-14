####################################################### 
### BASE SYSTEM SETUP FROM kdp:minimal DOCKER IMAGE ###
#######################################################

# starting from KDP-minimal v1.8.9
FROM ghcr.io/openkim/developer-platform:v1.8.9-minimal

# use ROOT user to install systemwide colabfit-mcp requirements
USER root 

# colabfit sys installations
RUN apt-get update && apt-get install -y \
    libgomp1 \
    nano \
    && rm -rf /var/lib/apt/lists/*

# this upgrades pip-26 much like kdp
# kdp must be doing this at a later stage probably because kdp-sys is only at pip-22
RUN python -m pip install --user --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel \
    build

# Good default envs
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1


###################################################### 
##### CLONE SOURCE CODE colabfit-mcp FROM GITHUB ##### 
###################################################### 


# colabfit-mcp repo url
ARG REPO_URL=https://github.com/colabfit/colabfit-mcp.git
# USE main BRANCH WHEN READY FOR MERGE
ARG REPO_REF=kdp_minimal_base

# use KDP packages directory
ARG PACKAGE_DIR=/pipeline/packages/ 

# # Clone repo into container
RUN git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} ${PACKAGE_DIR}/colabfit-mcp

# THIS IS PURELY FOR DEBUGGING!!!! DONT COMMIT THIS!!!
# COPY . /pipeline/packages/colabfit-mcp

# set working directory to /pipeline/packages
WORKDIR ${PACKAGE_DIR}/colabfit-mcp


##############################################$###
########## BUILD colabfit:minimal IMAGE ########## 
##############################################$###


# install packages listed in ${PACKAGE_DIR}/colabfit-mcp/pyproject.toml
RUN python -m pip install --no-cache-dir "mcp[cli]>=1.0.0,<2"
RUN python -m pip install --no-cache-dir "requests==2.33.1"
RUN python -m pip install --no-cache-dir "numpy==2.0.0"
RUN python -m pip install --no-cache-dir "pyarrow==24.0.0"
RUN python -m pip install --no-cache-dir "ase==3.25.0"
RUN python -m pip install --no-cache-dir "datasets>=2.0"
RUN python -m pip install --no-cache-dir "kliff @ git+https://github.com/gpwolfe/kliff.git@colabfit_mcp_edit"

# install colabfit-mcp source code
RUN pip install --no-cache-dir .

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import colabfit_mcp; print('ok')" || exit 1

# define ENTRYPOINT
ENTRYPOINT ["colabfit-mcp"]
    # this is crucial so that merely starting the container will fire up the server
    # as oppsed to firing up the server via an agent.py file

