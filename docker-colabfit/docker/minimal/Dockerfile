# start from the earlier built colabfit-git image as base
FROM colabfit-git:latest

# this is probably redundant. 
USER openkim

# use KDP packages directory
ARG PACKAGE_DIR=/pipeline/packages/ 

# set working directory to /pipeline/packages
WORKDIR ${PACKAGE_DIR}/colabfit-mcp


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


# install openai-agent library. This probably doesnt belong here. This is client-side code. Not server code. 
# RUN python -m pip install --no-cache-dir openai-agents

# install packages listed in ${PACKAGE_DIR}/colabfit-mcp/pyproject.toml
RUN python -m pip install --no-cache-dir "mcp[cli]>=1.0.0"
RUN python -m pip install --no-cache-dir "requests==2.33.1"
RUN python -m pip install --no-cache-dir "numpy==2.0.0"
RUN python -m pip install --no-cache-dir "pyarrow==24.0.0"
RUN python -m pip install --no-cache-dir "ase==3.25.0"
RUN python -m pip install --no-cache-dir "datasets>=2.0"
RUN python -m pip install --no-cache-dir "kliff @ git+https://github.com/gpwolfe/kliff.git@colabfit_mcp_edit"

# install colabfit-mcp source code
RUN pip install --user --no-cache-dir .

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import colabfit_mcp; print('ok')" || exit 1


# define ENTRYPOINT
ENTRYPOINT ["/home/openkim/.local/bin/colabfit-mcp"]
    # this is crucial so that merely starting the container will fire up the server
    # as oppsed to firing up the server via an agent.py file


# TO DO 
    # modify the pyproject.toml file 
    # the above pip install datasets and kliff should
    # be in the pyproject.toml [project] definition

