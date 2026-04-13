#!/bin/bash
set -e

IMAGE_NAME="nifty_trading_system"

echo "Building Podman image '${IMAGE_NAME}'..."
podman build -t $IMAGE_NAME -f Containerfile .

# Ensure host directories exist to mount
mkdir -p data models/saved output

echo "Running Podman container with arguments: $@"
# Volumes mapped so data/models persist on the host instead of dying with the container
podman run -it --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models/saved:/app/models/saved" \
  -v "$(pwd)/output:/app/output" \
  $IMAGE_NAME "$@"
