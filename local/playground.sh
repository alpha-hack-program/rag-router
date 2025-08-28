#!/bin/bash

export LLAMA_STACK_ENDPOINT=http://host.containers.internal:8080
# export LLAMA_STACK_ENDPOINT=https://lsd-rag-base.apps.ocp.sandbox425.opentlc.com

podman run -p 8501:8501 \
  --env NO_PROXY=localhost,127.0.0.1 \
  --env LLAMA_STACK_ENDPOINT=${LLAMA_STACK_ENDPOINT} \
  quay.io/rh-aiservices-bu/llama-stack-playground:0.2.11