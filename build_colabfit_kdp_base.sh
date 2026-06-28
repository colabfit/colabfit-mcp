docker build   -f docker-colabfit/docker/minimal/Dockerfile   -t colabfit-minimal   .;
docker build   -f docker-colabfit/docker/torchml/Dockerfile   -t colabfit-torchml   .;

# start the container with port-forwarding
# docker run -it -p 8000:8000 colabfit-minimal:latest bash