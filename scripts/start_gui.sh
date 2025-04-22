#!/bin/bash

# current folder as WORD_DIR
CURRENT_DIR=$(pwd)

set -eo pipefail

echo "using SIM_REPO_ROOT='$CURRENT_DIR'"
if [ -z "$SIM_ASSETS" ]; then
    echo "You need to set \$SIM_ASSETS eg. SIM_ASSETS=~/assets"
    exit 1
else
    echo "using SIM_ASSETS='$SIM_ASSETS'"
fi

xhost +
docker run -itd --name genie_sim_benchmark \
    --entrypoint ./scripts/entrypoint.sh \
    --gpus all \
    --rm \
    --network=host \
    --privileged \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -e "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python" \
    -e DISPLAY \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    -v /dev/input:/dev/input:rw \
    -v $SIM_ASSETS:/root/assets:rw \
    -v $CURRENT_DIR:/root/workspace/main:rw \
    -w /root/workspace/main \
    registry.agibot.com/genie-sim/open_source:latest
