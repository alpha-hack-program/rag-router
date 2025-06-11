#!/bin/bash

# If .test.env exists, source it
if [ -f .test.env ]; then
  source .test.env
else
  echo ".test.env file not found. Please create it with the necessary environment variables."
  exit 1
fi

MODEL_MAP_PATH=$(pwd)/scratch/model_map.json
# Fail if MODEL_MAP_PATH file does not exist
if [ ! -f "${MODEL_MAP_PATH}" ]; then
  echo "Model map file not found at ${MODEL_MAP_PATH}. Please ensure it exists."
  exit 1
fi

EMBEDDING_MAP_PATH=$(pwd)/scratch/embedding_map.json
# Fail if EMBEDDING_MAP_PATH file does not exist
if [ ! -f "${EMBEDDING_MAP_PATH}" ]; then
  echo "Embeddings map file not found at ${EMBEDDING_MAP_PATH}. Please ensure it exists."
  exit 1
fi

EMBEDDINGS_DEFAULT_MODEL="multilingual-e5-large-gpu"

# Echo the environment variables to verify they are set
echo "Using MODEL_MAP_PATH: ${MODEL_MAP_PATH}"
echo "Using EMBEDDING_MAP_PATH: ${EMBEDDING_MAP_PATH}"
echo "Using OpenAI Embedding API Key: ${OPENAI_EMBEDDING_API_KEY}"
echo "Using OpenAI Embedding URL: ${OPENAI_EMBEDDING_URL}"
echo "Using OpenAI Embedding Model: ${OPENAI_EMBEDDING_MODEL}"

# Echo Milvus connection details, host, port, username, password, collection name and database name
echo "Using Milvus Host: ${MILVUS_HOST}"
echo "Using Milvus Port: ${MILVUS_PORT}"
echo "Using Milvus Username: ${MILVUS_USERNAME}"
echo "Using Milvus Password: ${MILVUS_PASSWORD}"
echo "Using Milvus Database Name: ${MILVUS_DATABASE_NAME}"
echo "Using Milvus Collection Name: ${MILVUS_COLLECTION_NAME}"

# Export the environment variables
export MODEL_MAP_PATH EMBEDDING_MAP_PATH EMBEDDINGS_DEFAULT_MODEL
export OPENAI_EMBEDDING_API_KEY OPENAI_EMBEDDING_URL OPENAI_EMBEDDING_MODEL
export MILVUS_HOST MILVUS_PORT MILVUS_USERNAME MILVUS_PASSWORD
export MILVUS_DATABASE_NAME MILVUS_COLLECTION_NAME

# Run the Python script
export LOG_LEVEL=DEBUG

uvicorn main:app --log-level debug --host 0.0.0.0 --port 7777 --reload
if [ $? -ne 0 ]; then
  echo "Python script execution failed."
  exit 1
else
  echo "Python script executed successfully."
fi