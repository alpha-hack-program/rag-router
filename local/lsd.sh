#!/bin/bash

DISTRIBUTION_IMAGE=quay.io/opendatahub/llama-stack:odh

export LLAMA_STACK_PORT=8080
export LLAMA_STACK_SERVER=http://localhost:$LLAMA_STACK_PORT

if [ ! -f .env ]; then
  echo ".env file not found. Please create it with the necessary environment variables."
  exit 1
fi

source .env

# Curl to list the models
curl -s -X GET -H "Authorization: Bearer ${GRANITE_3_3_8B_API_TOKEN}" ${GRANITE_3_3_8B_URL}/models | jq .
curl -s -X GET -H "Authorization: Bearer ${LLAMA_3_1_8B_W4A16_API_TOKEN}" ${LLAMA_3_1_8B_W4A16_URL}/models | jq .

# Curl to test the models
curl -s -X POST -H "Content-Type: application/json" -H "Authorization: Bearer ${GRANITE_3_3_8B_API_TOKEN}" -d '{"messages": [{"role": "user", "content": "Hello, how are you?"}], "model": "'${GRANITE_3_3_8B_MODEL}'"}' ${GRANITE_3_3_8B_URL}/chat/completions | jq .
curl -s -X POST -H "Content-Type: application/json" -H "Authorization: Bearer ${LLAMA_3_1_8B_W4A16_API_TOKEN}" -d '{"messages": [{"role": "user", "content": "Hello, how are you?"}], "model": "'${LLAMA_3_1_8B_W4A16_MODEL}'"}' ${LLAMA_3_1_8B_W4A16_URL}/chat/completions | jq .

# Run the container
podman run -it --rm \
  --name llama-stack \
  -p ${LLAMA_STACK_PORT}:${LLAMA_STACK_PORT} \
  -e NO_PROXY=localhost,127.0.0.1 \
  -e "MILVUS_DB_PATH=/opt/app-root/.milvus/milvus.db" \
  -e "FMS_ORCHESTRATOR_URL=http://localhost" \
  -e "GRANITE_3_3_8B_URL=${GRANITE_3_3_8B_URL}" \
  -e "GRANITE_3_3_8B_TLS_VERIFY=${GRANITE_3_3_8B_TLS_VERIFY}" \
  -e "GRANITE_3_3_8B_API_TOKEN=${GRANITE_3_3_8B_API_TOKEN}" \
  -e "GRANITE_3_3_8B_MAX_TOKENS=${GRANITE_3_3_8B_MAX_TOKENS}" \
  -e "GRANITE_3_3_8B_MODEL=${GRANITE_3_3_8B_MODEL}" \
  -e "LLAMA_3_1_8B_W4A16_URL=${LLAMA_3_1_8B_W4A16_URL}" \
  -e "LLAMA_3_1_8B_W4A16_TLS_VERIFY=${LLAMA_3_1_8B_W4A16_TLS_VERIFY}" \
  -e "LLAMA_3_1_8B_W4A16_API_TOKEN=${LLAMA_3_1_8B_W4A16_API_TOKEN}" \
  -e "LLAMA_3_1_8B_W4A16_MAX_TOKENS=${LLAMA_3_1_8B_W4A16_MAX_TOKENS}" \
  -e "LLAMA_3_1_8B_W4A16_MODEL=${LLAMA_3_1_8B_W4A16_MODEL}" \
  -e HF_HOME=/cache/huggingface \
  -e "NAMESPACE=llama-stack" \
  -e "KUBECONFIG=/opt/app-root/src/.kube/config" \
  -v $(pwd)/.hf_cache:/cache/huggingface:Z \
  -v $(pwd)/.milvus:/opt/app-root/.milvus:Z \
  -v $(pwd)/run.yaml:/opt/app-root/run.yaml:ro \
  -v ~/.kube:/opt/app-root/src/.kube:ro \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp:/tmp \
  $DISTRIBUTION_IMAGE --port ${LLAMA_STACK_PORT}

# DISTRIBUTION_IMAGE=llamastack/distribution-ollama:0.2.2

# export LLAMA_STACK_MODEL="llama3.2:3b"
# export INFERENCE_MODEL="llama3.2:3b"
# export LLAMA_STACK_PORT=8321
# export LLAMA_STACK_SERVER=http://localhost:$LLAMA_STACK_PORT

# podman run -it \
#   -p ${LLAMA_STACK_PORT}:${LLAMA_STACK_PORT} \
#   ${DISTRIBUTION_IMAGE} \
#   --port ${LLAMA_STACK_PORT} \
#   --env NO_PROXY=localhost,127.0.0.1 \
#   --env INFERENCE_MODEL=${LLAMA_STACK_MODEL} \
#   --env OLLAMA_URL=http://host.containers.internal:11434