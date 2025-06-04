#!/usr/bin/env bash

set -euo pipefail

# Source env vars
if [[ -f .test.env ]]; then
  source .test.env
else
  echo ".test.env not found."
  exit 1
fi

# Query input
if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <your question>"
  exit 1
fi

QUERY="$*"
HOST_IP=localhost
RAG_ROUTER_URL="http://${HOST_IP}:7777/v1/chat/completions"

CHAT_MODEL=granite-3-1-8b
# CHAT_MODEL=mistral-7b-instruct-v0-3
# CHAT_MODEL=llama-3-1-8b

echo "== Non-streaming =="
curl -sS -X POST "${RAG_ROUTER_URL}" \
  -H "Content-Type: application/json" \
  -H "x-db-type: milvus" \
  -d @<(cat <<EOF
{
  "model": "${CHAT_MODEL}",
  "messages": [
    { "role": "user", "content": "${QUERY//\"/\\\"}" }
  ],
  "temperature": 0.7,
  "top_p": 1.0,
  "max_tokens": 200,
  "stream": false
}
EOF
)

echo -e "\n\n== Streaming =="
curl -sS -N -X POST "$RAG_ROUTER_URL" \
  -H "Content-Type: application/json" \
  -H "x-db-type: milvus" \
  -d @<(cat <<EOF
{
  "model": "${CHAT_MODEL}",
  "messages": [
    { "role": "user", "content": "${QUERY//\"/\\\"}" }
  ],
  "temperature": 0.7,
  "top_p": 1.0,
  "max_tokens": 200,
  "stream": true
}
EOF
)
