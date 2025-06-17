import os
import sys
import logging
import json
from token import OP

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from vectorizer import get_vector
from retriever import retrieve_context
from prompt_builder import build_prompt
from llm_router import stream_openai_chat, complete_openai_chat
from pymilvus import connections, utility
from contextlib import asynccontextmanager

# Allowed log levels
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

# Read from environment variable
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "WARNING").upper()

# --- Chat-prompt configuration ---------------------------------------------
BASE_SYSTEM_PROMPT = (
    "You are a retrieval-augmented generation (RAG) assistant. "
    "Answer the user’s question as helpfully as possible. "
    "Use the information inside <doc> … </doc> tags when relevant; "
    "if the context is insufficient, say you don’t know."
)

def configure_logging():
    """Ensure logging is configured early and correctly."""
    if LOG_LEVEL_STR not in VALID_LOG_LEVELS:
        raise ValueError(f"Invalid log level: {LOG_LEVEL_STR}. Must be one of {VALID_LOG_LEVELS}.")

    root_logger = logging.getLogger()
    if not root_logger.hasHandlers():  # Avoid reconfiguration if already set
        logging.basicConfig(
            level=LOG_LEVEL_STR,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stdout,
        )

    # Optional: force override Uvicorn’s default level
    root_logger.setLevel(LOG_LEVEL_STR)

    # Suppress noisy logs from specific libraries
    logging.getLogger('docling').setLevel(logging.ERROR)

# Run as early as possible
configure_logging()

# Logging instance for this module
_log = logging.getLogger(__name__)

# MINMUM COSINE DISTANCE
MINIMUM_COSINE_DISTANCE = float(os.getenv("MINIMUM_COSINE_DISTANCE", "0.85"))

# Load model config map from file
MODEL_URL_SUFFIX = os.getenv("MODEL_URL_SUFFIX", "v1/chat/completions")
MODEL_MAP_PATH = os.getenv("MODEL_MAP_PATH", "/etc/secrets/model_map.json")

try:
    with open(MODEL_MAP_PATH, "r") as f:
        MODEL_CONFIG_MAP = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model configuration from {MODEL_MAP_PATH}: {e}")

# Load embedding config map from file
EMBEDDING_MAP_PATH = os.getenv("EMBEDDING_MAP_PATH")
EMBEDDINGS_DEFAULT_MODEL = os.getenv("EMBEDDINGS_DEFAULT_MODEL")

# Check if EMBEDDING_MAP_PATH is set
if not EMBEDDING_MAP_PATH:
    raise RuntimeError("EMBEDDING_MAP_PATH must be set in environment variables.")
# Check if EMBEDDINGS_DEFAULT_MODEL is set
if not EMBEDDINGS_DEFAULT_MODEL:
    raise RuntimeError("EMBEDDINGS_DEFAULT_MODEL must be set in environment variables.")

# Load the openai key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

try:
    with open(EMBEDDING_MAP_PATH, "r") as f:
        EMBEDDING_CONFIG_MAP = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load embedding configuration from {EMBEDDING_MAP_PATH}: {e}")

embedding_config = EMBEDDING_CONFIG_MAP.get(EMBEDDINGS_DEFAULT_MODEL)
if embedding_config is None:
    raise RuntimeError(f"Embedding model '{EMBEDDINGS_DEFAULT_MODEL}' not found in embeddings map.")

EMBEDDING_URL_SUFFIX = embedding_config.get("url_suffix", "v1/embeddings")
OPENAI_EMBEDDING_API_KEY = embedding_config.get("api_key", "")
OPENAI_EMBEDDING_URL = embedding_config.get("url", "")
OPENAI_EMBEDDING_MODEL = embedding_config.get("model", "")

if not OPENAI_EMBEDDING_URL or not OPENAI_EMBEDDING_MODEL:
    raise RuntimeError("Incomplete embedding model configuration.")

# Environment variables for Milvus configuration
MILVUS_USERNAME = os.getenv("MILVUS_USERNAME", "")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
MILVUS_DATABASE = os.getenv("MILVUS_DATABASE", "default")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "documents")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
try:
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
except ValueError:
    raise RuntimeError(f"Invalid MILVUS_PORT value: {os.getenv('MILVUS_PORT')}. It must be an integer.")


# Runtime error handling for Milvus configuration
if not MILVUS_HOST or not MILVUS_PORT:
    raise RuntimeError("Milvus host and port must be set in environment variables.")
if not MILVUS_USERNAME or not MILVUS_PASSWORD:
    raise RuntimeError("Milvus username and password must be set in environment variables.")

_log.debug(">>> Module loaded and logger configured.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    _log.info(f"Connecting to Milvus => {MILVUS_HOST}:{MILVUS_PORT} with collection '{MILVUS_COLLECTION_NAME}' and database '{MILVUS_DATABASE}'...")
    connections.connect(alias="default", 
                        host=MILVUS_HOST, port=MILVUS_PORT, 
                        user=MILVUS_USERNAME, password=MILVUS_PASSWORD, db_name=MILVUS_DATABASE)
    if not utility.has_collection(MILVUS_COLLECTION_NAME):
        raise RuntimeError("Milvus collection does not exist.")
    _log.info("Connected to Milvus.")
    try:
        yield
    finally:
        _log.info("Disconnecting from Milvus...")
        connections.disconnect(alias="default")
        _log.info("Disconnected.")

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def debug_requests(request: Request, call_next):
    # Log detailed request information
    _log.info(f"=== INCOMING REQUEST ===")
    _log.info(f"Method: {request.method}")
    _log.info(f"URL: {request.url}")
    _log.info(f"Client: {request.client}")
    _log.info(f"Headers: {dict(request.headers)}")
    _log.info(f"Path: {request.url.path}")
    _log.info(f"Query params: {request.query_params}")
    
    try:
        # Process the request
        response = await call_next(request)
        _log.info(f"Response status: {response.status_code}")
        _log.info(f"Response headers: {dict(response.headers)}")
        return response
    except Exception as e:
        _log.error(f"Exception during request processing: {e}")
        _log.error(f"Exception type: {type(e)}")
        import traceback
        _log.error(f"Traceback: {traceback.format_exc()}")
        raise
        
@app.post("/v1/chat/completions")
async def openai_chat_completions(
    request: Request,
    x_db_type: str = Header(default="milvus")
):
    _log.debug("DEBUG: entering openai_chat_completions")
    _log.info("INFO: entering openai_chat_completions")

    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    if not body:
        raise HTTPException(status_code=400, detail="Request body cannot be empty.")

    _log.debug(f"Received request body: {body}")

    # Extract the model
    model_name = body.get("model")
    if model_name is None:
        raise HTTPException(status_code=400, detail="`model` is required in the request body.")

    model_config = MODEL_CONFIG_MAP.get(model_name)
    if model_config is None:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name}")

    chat_url = f'{model_config.get("url")}/{MODEL_URL_SUFFIX}'
    chat_model = model_config.get("model")
    chat_api_key = model_config.get("api_key")

    if not chat_url or not chat_model:
        raise HTTPException(status_code=500, detail="Model configuration is incomplete or missing.")

    _log.debug(f"Using model: {chat_model} at {chat_url}")
    
    # Extract parameters from the request body
    messages = body.get("messages", [])
    temperature = body.get("temperature", 1.0)
    top_p = body.get("top_p", 1.0)
    max_tokens = body.get("max_tokens", 512)
    stream = body.get("stream", True)
    openai_key = body.get("openai_key")

    # Check the provided OpenAI key is vald against the configured key
    if OPENAI_API_KEY and openai_key != OPENAI_API_KEY:
        _log.warning("Provided OpenAI key does not match the configured key.")
        raise HTTPException(status_code=403, detail="Invalid OpenAI API key.")
    
    # Validate the messages structure
    query_text = messages[-1]["content"] if messages else ""

    _log.debug(f"Received query: {query_text}")

    # Generate the query vector using OpenAI's embedding API
    openai_embedding_url = f'{OPENAI_EMBEDDING_URL}/{EMBEDDING_URL_SUFFIX}'
    query_vector = await get_vector(
        text=query_text,
        url=openai_embedding_url,
        model=OPENAI_EMBEDDING_MODEL,
        openai_key=OPENAI_EMBEDDING_API_KEY
    )
    if not query_vector:
        raise HTTPException(status_code=500, detail="Failed to generate query vector.")

    _log.debug(f"Generated query vector of length: {len(query_vector)}")
    
    # Retrieve context from the database, it should have content, source and headings
    documents = await retrieve_context(query_vector, db_type=x_db_type)

    if not documents:
        _log.warning("No documents found for the given query vector.")

    _log.debug(f"Retrieved documents: {documents}")

    # Use only document with cosine distance > 0.8
    documents = [doc for doc in documents if doc.get("distance", 0) > MINIMUM_COSINE_DISTANCE]

    # Build the prompt for the LLM
    prompt = build_prompt(query_text, [doc["content"] for doc in documents])

    _log.debug(f"Built prompt: {prompt}")

    # For each document extract references as: source -> heading1 -> heading2
    # headings is a single string with lines separated by newlines
    references = []
    for doc in documents:
        source = doc.get("source", "Unknown Source")
        headings = doc.get("headings", "")
        if headings:
            headings_list = [h.strip() for h in headings.split("\n") if h.strip()]
            if len(headings_list) == 1:
                references.append(f"{source} -> {headings_list[0]}")
            elif len(headings_list) > 1:
                references.append(f"{source} -> " + " -> ".join(headings_list))
            else:
                references.append(source)
        else:
            references.append(source)
    _log.debug(f"References extracted: {references}")
    
    # If the request is for streaming, use the streaming endpoint
    if stream:
        return StreamingResponse(
            stream_openai_chat(
                prompt=prompt,
                model=chat_model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                url=chat_url,
                openai_key=chat_api_key,
                references=references
            ),
            media_type="text/event-stream"
        )
    else:
        content = await complete_openai_chat(
            prompt=prompt,
            model=chat_model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            url=chat_url,
            openai_key=chat_api_key,
            references=references
        )
        return JSONResponse(content={
            "id": "chatcmpl-mocked-id",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }]
        })

@app.get("/v1/models")
async def list_models():
    _log.debug("Listing available models.")
    return JSONResponse(content={
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "owned_by": "local"
            }
            for model_name in MODEL_CONFIG_MAP.keys()
        ]
    })

@app.get("/test")
async def test_endpoint():
    return {"message": "Hello from external IP test", "status": "success"}