# starting from KDP-minimal
FROM ghcr.io/openkim/developer-platform:latest-minimal

# temporarily switch to ROOT user to install systemwide colabfit-mcp requirements
USER root 

# colabfit sys installations
RUN apt-get update && apt-get install -y \
    libgomp1 \
    nano \
    && rm -rf /var/lib/apt/lists/*

# revert back to openkim user
USER openkim


# this upgrades pip-26 much like kdp
# kdp must be doing this at a later stage probably because kdp-sys is only at pip-22
RUN python -m pip install --user --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel \
    build

    
# Default working directory
WORKDIR /home/openkim

# Good default envs
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1