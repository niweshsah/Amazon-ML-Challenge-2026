#!/bin/bash

CONTAINER_NAME="isaac-sim"

# Check if the container is currently running
if [ "$(docker ps -q -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Container '${CONTAINER_NAME}' is already running. Opening a new bash session inside it..."
    # Use exec instead of attach so exiting this shell doesn't kill the main container
    docker exec -it ${CONTAINER_NAME} bash
else
    echo "Starting a new '${CONTAINER_NAME}' container..."
    docker run --name ${CONTAINER_NAME} --entrypoint bash -it --gpus all \
        -e "ACCEPT_EULA=Y" \
        --rm \
        --network=host \
        -e "PRIVACY_CONSENT=Y" \
        -v ~/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw \
        -v ~/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
        -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
        -v ~/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
        -v ~/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw \
        -v ~/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw \
        -v ~/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw \
        -v ~/.cache/ov/hub:/var/cache/hub:rw \
        -u 1234:1234 \
        nvcr.io/nvidia/isaac-sim:6.1.0
fi