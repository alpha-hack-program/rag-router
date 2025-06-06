#!/bin/bash

set -euo pipefail

# --- Load .env (optional, for build-time defaults) ---
if [ -f .env ]; then
  source .env
fi

# --- Default values ---
COMPONENT_NAME="${COMPONENT_NAME:-$DEFAULT_COMPONENT_NAME}"
TAG="${TAG:-$DEFAULT_TAG}"
BASE_IMAGE="${BASE_IMAGE:-$DEFAULT_BASE_IMAGE}"
BASE_TAG="${BASE_TAG:-$DEFAULT_BASE_TAG}"
CONTAINER_FILE="${CONTAINER_FILE:-$DEFAULT_CONTAINER_FILE}"
CACHE_FLAG="${CACHE_FLAG:-$DEFAULT_CACHE_FLAG}"
PORT="${PORT:-$DEFAULT_PORT}"
REGISTRY="${REGISTRY:-$DEFAULT_REGISTRY}"

LOCAL_IMAGE="${COMPONENT_NAME}:${TAG}"
REMOTE_IMAGE="${REGISTRY}/${COMPONENT_NAME}:${TAG}"

# --- Build the image ---
function build() {
  echo "🔨 Building image ${LOCAL_IMAGE} with base image ${BASE_IMAGE}:${BASE_TAG}"
  podman build ${CACHE_FLAG} \
    -t "${LOCAL_IMAGE}" \
    -f "${CONTAINER_FILE}" . \
    --build-arg BASE_IMAGE="${BASE_IMAGE}:${BASE_TAG}" \
    --build-arg COMPONENT_NAME="${COMPONENT_NAME}"
  echo "✅ Build complete: ${LOCAL_IMAGE}"
}

# --- Push the image to registry ---
function push() {
  echo "📤 Pushing image to ${REMOTE_IMAGE}..."
  podman tag "${LOCAL_IMAGE}" "${REMOTE_IMAGE}"
  podman push "${REMOTE_IMAGE}"
  echo "✅ Image pushed to: ${REMOTE_IMAGE}"
}

# --- Run the image ---
function run() {
  USE_REMOTE=false

  if [[ "${1:-}" == "--remote" ]]; then
    USE_REMOTE=true
    shift
  fi

  IMAGE_TO_RUN="${LOCAL_IMAGE}"
  if $USE_REMOTE; then
    echo "🌐 Pulling remote image ${REMOTE_IMAGE}..."
    podman pull "${REMOTE_IMAGE}"
    IMAGE_TO_RUN="${REMOTE_IMAGE}"
  else
    echo "🚀 Running local image ${LOCAL_IMAGE} on port ${PORT}..."
  fi

  # Pick env file
  if [ -f .test.env ]; then
    ENV_FILE=".test.env"
  else
    echo "❌ No .test.env file found."
    exit 1
  fi

  echo "📄 Using environment file: ${ENV_FILE}"

  # Extract MODEL_MAP_PATH from env file robustly
  MODEL_MAP_PATH_IN_CONTAINER_RAW=$(grep -E '^\s*MODEL_MAP_PATH\s*=' "$ENV_FILE" | sed -E 's/^\s*MODEL_MAP_PATH\s*=\s*["'\''"]?([^"'\''"]*)["'\''"]?.*/\1/')

  # Resolve relative container path to absolute if needed
  if [[ "$MODEL_MAP_PATH_IN_CONTAINER_RAW" = /* ]]; then
    MODEL_MAP_PATH_IN_CONTAINER="$MODEL_MAP_PATH_IN_CONTAINER_RAW"
  else
    MODEL_MAP_PATH_IN_CONTAINER="/app/$MODEL_MAP_PATH_IN_CONTAINER_RAW"
  fi

  # Check if MODEL_MAP_PATH_IN_CONTAINER is set
  if [ -z "$MODEL_MAP_PATH_IN_CONTAINER" ]; then
    echo "❌ MODEL_MAP_PATH is not defined in ${ENV_FILE}"
    exit 1
  fi

  MODEL_MAP_PATH_ON_HOST="$(pwd)/scratch/model_map.json"

  if [ ! -f "$MODEL_MAP_PATH_ON_HOST" ]; then
    echo "❌ Model map file not found at ${MODEL_MAP_PATH_ON_HOST}"
    exit 1
  fi

  echo "📁 Mounting model map:"
  echo "    Host:      ${MODEL_MAP_PATH_ON_HOST}"
  echo "    Container: ${MODEL_MAP_PATH_IN_CONTAINER}"

  podman run --rm -it --net=host --name rag-router \
    --env-file "${ENV_FILE}" \
    -p "${PORT}:${PORT}" \
    -v "${MODEL_MAP_PATH_ON_HOST}:${MODEL_MAP_PATH_IN_CONTAINER}:ro" \
    "${IMAGE_TO_RUN}"
}

# --- Show usage ---
function help() {
  echo "Usage: ./image.sh [build|push|run [--remote]|all]"
  echo "  build         Build the container image"
  echo "  push          Push the image to the registry"
  echo "  run           Run the local image"
  echo "  run --remote  Run the image pulled from the registry"
  echo "  all           Build, push, and run locally"
}

# --- Entrypoint ---
case "${1:-}" in
  build) build ;;
  push) push ;;
  run) shift; run "$@" ;;
  all)
    build
    push
    run
    ;;
  *) help ;;
esac
