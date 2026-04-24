# start from earlier built colabfit-sys image as base
FROM colabfit-sys:latest

# colabfit-mcp repo url
ARG REPO_URL=https://github.com/colabfit/colabfit-mcp.git
# USE main BRANCH WHEN READY FOR MERGE
ARG REPO_REF=kdp_minimal_base

# use KDP packages directory
ARG PACKAGE_DIR=/pipeline/packages/ 

# Clone repo into container
RUN git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} ${PACKAGE_DIR}/colabfit-mcp

# set working directory to /pipeline/packages
WORKDIR ${PACKAGE_DIR}/colabfit-mcp