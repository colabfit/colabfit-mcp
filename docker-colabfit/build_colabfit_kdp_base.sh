docker build   -f /home/nmohan/UMN/colabfit-mcp/docker-colabfit/docker/minimal/Dockerfile   -t colabfit-minimal   /home/nmohan/UMN/colabfit-mcp;
docker build   -f /home/nmohan/UMN/colabfit-mcp/docker-colabfit/docker/torchml/Dockerfile   -t colabfit-torchml   /home/nmohan/UMN/colabfit-mcp;

# start the container with port-forwarding
# docker run -it -p 8000:8000 colabfit-minimal:latest bash