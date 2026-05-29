# build minimal image (for search/download)
	docker build   -f /home/nmohan/UMN/colabfit-mcp/docker-colabfit/docker/minimal/Dockerfile   -t colabfit-minimal   /home/nmohan/UMN/colabfit-mcp;

# build torchml-image (for training)
	# docker build   -f /home/nmohan/UMN/colabfit-mcp/docker-colabfit/docker/torchml/Dockerfile   -t colabfit-torchml   /home/nmohan/UMN/colabfit-mcp;

# start the container with port-forwarding
	# docker run -it -p 8000:8000 colabfit-minimal:latest bash

# to start a singularity container - you dont need port-forwarding in singularity
	# singularity exec colabfit-minimal.sif colabfit-mcp
