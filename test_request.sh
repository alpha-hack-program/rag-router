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
ROUTER_HOST=${ROUTER_HOST:-http://localhost:7856}
echo "Using RAG router at: ${ROUTER_HOST}"
RAG_ROUTER_URL="${ROUTER_HOST}/v1/chat/completions"
echo "RAG Router URL: ${RAG_ROUTER_URL}"
echo "Query: ${QUERY}"

# CHAT_MODEL=granite-3-3-8b
# CHAT_MODEL=mistral-7b-instruct-v0-3
CHAT_MODEL=llama-3-1-8b-w4a16

# If environment variable USE_STREAMING is set, use streaming
if [[ "${USE_STREAMING:-false}" == "true" ]]; then
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
else
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
fi




