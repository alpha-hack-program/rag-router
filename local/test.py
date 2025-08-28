import os

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from llama_stack_client import LlamaStackClient
from llama_stack_client.types.model import Model
from llama_stack_client.types.shared_params.document import Document as RAGDocument
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger as AgentEventLogger


def setup_environment() -> None:
    """Setup environment variables with defaults"""
    os.environ["INFERENCE_MODEL"] = "llama-4-scout-17b-16e-w4a16"
    # os.environ["INFERENCE_MODEL"] = "granite-3-3-8b-instruct"
    os.environ["INFERENCE_MODEL_PROVIDER"] = "vllm-inference"
    
    os.environ["EMBEDDING_MODEL"] = "granite-embedding-125m"
    os.environ["EMBEDDING_DIMENSION"] = "768"
    os.environ["EMBEDDING_MODEL_PROVIDER"] = "sentence-transformers"
    # os.environ["EMBEDDING_MODEL"] = "multilingual-e5-large-gpu"
    # os.environ["EMBEDDING_DIMENSION"] = "1024"
    # os.environ["EMBEDDING_MODEL_PROVIDER"] = "vllm-inference-embedding"
    
    os.environ["BON_CALCULADORA_TOOLGROUP_ID"] = "mcp::bon-calculadora"
    os.environ["BON_CALCULADORA_MCP_ENDPOINT"] = "http://host.containers.internal:8000/sse"

    os.environ["LLAMA_STACK_HOST"] = "localhost"
    os.environ["LLAMA_STACK_PORT"] = "8080"
    os.environ["LLAMA_STACK_SECURE"] = "False"
    # os.environ["LLAMA_STACK_HOST"] = "lsd-rag-base.apps.ocp.sandbox425.opentlc.com"
    # os.environ["LLAMA_STACK_PORT"] = "443"
    # os.environ["LLAMA_STACK_SECURE"] = "True"


def create_client(host: str, port: int, secure: bool = False) -> LlamaStackClient:
    """Initialize and return the LlamaStack client"""
    if secure:
        protocol: str = "https"
    else:
        protocol: str = "http"

    if not (1 <= port <= 65535):
        raise ValueError(f"Port number {port} is out of valid range (1-65535).")
    if not host:
        raise ValueError("Host must be specified and cannot be empty.")
    
    return LlamaStackClient(base_url=f"{protocol}://{host}:{port}")

def get_embedding_model(
    client: LlamaStackClient,
    embedding_model_id: str,
    embedding_model_provider: str
) -> Model:
    """Fetch and return the embedding model by ID and provider"""
    if not embedding_model_id:
        raise ValueError("Embedding model ID is required")
    if not embedding_model_provider:
        raise ValueError("Embedding model provider is required")
    
    models = client.models.list()
    for model in models:
        if model.identifier == embedding_model_id and model.provider_id == embedding_model_provider and model.api_model_type == "embedding":
            return model
    
    raise ValueError(f"Embedding model {embedding_model_id} not found for provider {embedding_model_provider}")

def register_mcp_toolgroup(
    client: LlamaStackClient, 
    toolgroup_name: str,
    mcp_endpoint_uri: str
) -> str:
    """Register the MCP toolgroup and return the toolgroup identifier"""
    toolgroup_id = f"mcp::{toolgroup_name}"
    
    try:
        # Register the MCP toolgroup using the toolgroups API
        client.toolgroups.register(
            provider_id="model-context-protocol",
            toolgroup_id=toolgroup_id,
            mcp_endpoint={
                "uri": mcp_endpoint_uri
            }
        )
        print(f"Registered MCP toolgroup: {toolgroup_id}")
        return toolgroup_id
        
    except Exception as e:
        print(f"Error registering MCP toolgroup {toolgroup_id}: {e}")
        # If toolgroup already exists, that's usually fine
        if "already exists" in str(e).lower() or "already registered" in str(e).lower():
            print(f"Toolgroup {toolgroup_id} already registered")
            return toolgroup_id
        raise

def get_models(client: LlamaStackClient) -> Tuple[str, Model]:
    """Fetch and display available models, return model IDs"""
    models = client.models.list()
    print(f"models = {models}")
    
    print("Available models:")
    for model in models:
        print(f"- {model.identifier} (type: {model.api_model_type}, provider: {model.provider_id})")
    
    model_id: str = next(m.identifier for m in models if m.api_model_type == "llm")
    embedding_model = next(m for m in models if m.api_model_type == "embedding")
    
    return model_id, embedding_model


def register_vector_db(
    client: LlamaStackClient, 
    embedding_model: Model, 
    vector_db_id: str = "milvus_bon_db", 
    provider_id: str = "milvus"
) -> str:
    """Register vector database"""
    if not vector_db_id:
        raise ValueError("Vector DB ID is required for registration")
    if not provider_id:
        raise ValueError("Provider ID is required for vector DB registration")
    
    if not embedding_model:
        raise ValueError("Embedding model is required for vector DB registration")
    
    embedding_model_id: str = embedding_model.identifier
    if not embedding_model_id:
        raise ValueError("Embedding model ID is required for vector DB registration")
    
    # Check it model api type is 'embedding'
    if embedding_model.api_model_type != "embedding":
        raise ValueError("Provided model is not an embedding model")

    # Check if embedding model metadata contains 'embedding_dimension' and if it's a str, int or float
    if not hasattr(embedding_model, 'metadata') or not isinstance(embedding_model.metadata, dict):
        raise ValueError("Embedding model metadata must be a dictionary")
    if not isinstance(embedding_model.metadata, dict):
        raise ValueError("Embedding model metadata must be a dictionary")
    if not isinstance(embedding_model.metadata.get("embedding_dimension"), (str, int, float)):
        raise ValueError("Embedding model metadata 'embedding_dimension' must be a str, int or float")
    
    embedding_dimension: int = embedding_model.metadata["embedding_dimension"] # type: ignore
    
    print(f"Registering vector DB: {vector_db_id} with embedding model {embedding_model_id} (dimension: {embedding_dimension})")
    client.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model=embedding_model_id,
        # embedding_dimension=embedding_dimension,
        provider_id=provider_id,
    )
    print(f"Registered vector DB: {vector_db_id}")
    return vector_db_id


def get_mime_type(extension: str) -> str:
    """Get MIME type based on file extension"""
    mime_types: dict[str, str] = {
        '.txt': 'text/plain',
        '.md': 'text/markdown', 
        '.py': 'text/plain',
        '.json': 'application/json',
        '.html': 'text/html',
        '.csv': 'text/csv'
    }
    return mime_types.get(extension.lower(), 'text/plain')


def load_documents_from_folder(
    folder_path: str, 
    file_extensions: List[str] = ['.txt', '.md']
) -> List[RAGDocument]:
    """Load documents from a local folder and return RAGDocument objects"""
    documents: List[RAGDocument] = []
    folder: Path = Path(folder_path)
    
    if not folder.exists():
        print(f"Warning: Folder {folder_path} does not exist")
        return documents
    
    print(f"Loading documents from: {folder_path}")
    
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in file_extensions:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content: str = f.read()
                
                mime_type: str = get_mime_type(file_path.suffix)
                
                doc: RAGDocument = RAGDocument(
                    document_id=file_path.stem,
                    content=content,
                    mime_type=mime_type,
                    metadata={
                        "filename": file_path.name,
                        "filepath": str(file_path),
                        "file_size": file_path.stat().st_size
                    }
                )
                documents.append(doc)
                print(f"Loaded: {file_path.name}")
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    print(f"Successfully loaded {len(documents)} documents")
    return documents


def insert_documents(
    client: LlamaStackClient, 
    documents: List[RAGDocument], 
    vector_db_id: str, 
    chunk_size_in_tokens: int = 512
) -> None:
    """Insert documents into the vector database"""
    if not documents:
        print("No documents to insert")
        return
    
    client.tool_runtime.rag_tool.insert(
        documents=documents,
        vector_db_id=vector_db_id,
        chunk_size_in_tokens=chunk_size_in_tokens,
    )
    print(f"Inserted {len(documents)} documents into vector DB")


def create_rag_agent_with_mcp(
    client: LlamaStackClient, 
    model_id: str, 
    tools: Optional[List[Union[str, Any]]] = None,
    instructions: str = "You are a helpful assistant"
) -> Agent:
    """Create and return a RAG agent with MCP toolgroup support"""
    print(f"Creating RAG agent with model: {model_id}")
    if not model_id:
        raise ValueError("Model ID is required to create the agent")
    if not tools:
        tools = []
    if not isinstance(tools, list):
        raise ValueError("Tools must be a list of tool identifiers or configurations")
    print(f"Using tools: {tools}")
    print(f"Agent instructions: {instructions}")
    return Agent(
        client,
        model=model_id,
        instructions=instructions,
        enable_session_persistence=False,
        tools=tools or [],
        tool_config={"tool_choice": "auto" if tools else "none"},
    )


def process_user_prompts(agent: Agent, prompts: List[str]) -> None:
    """Process a list of user prompts with the RAG agent"""
    session_id: str = agent.create_session("test-session")
    
    for prompt in prompts:
        print(f"User> {prompt}")
        response = agent.create_turn(
            messages=[{"role": "user", "content": prompt}],
            session_id=session_id,
        )
        for log in AgentEventLogger().log(response):
            log.print()


def main() -> None:
    """Main function to orchestrate the RAG setup and querying"""

    # There are two args: #1 mode (auto | rag | none) #2 (optional) no-insert
    import argparse
    parser = argparse.ArgumentParser(description="Run RAG agent with MCP tools")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["auto", "rag", "mcp", "none"], 
        default="auto",
        help="Mode to run the agent: 'auto' for automatic tool choice, 'rag' for RAG mode, 'mcp' for only tools and 'none' for no tools"
    )
    parser.add_argument(
        "--no-insert", 
        action="store_true", 
        help="If set, will not insert documents into the vector database"
    )
    args = parser.parse_args()
    print(f"Running in mode: {args.mode}, no-insert: {args.no_insert}")
 
    # Setup environment variables
    setup_environment()

    # Get environment variables
    model_id = os.environ.get("INFERENCE_MODEL")
    if model_id is None:
        raise ValueError("INFERENCE_MODEL environment variable must be set")
    model_provider = os.environ.get("INFERENCE_MODEL_PROVIDER")
    if model_provider is None:
        raise ValueError("INFERENCE_MODEL_PROVIDER environment variable must be set")

    embedding_model_id = os.environ.get("EMBEDDING_MODEL")
    if embedding_model_id is None:
        raise ValueError("EMBEDDING_MODEL environment variable must be set")
    embedding_model_dimension = os.environ.get("EMBEDDING_DIMENSION")
    if embedding_model_dimension is None:
        raise ValueError("EMBEDDING_DIMENSION environment variable must be set")
    embedding_model_provider = os.environ.get("EMBEDDING_MODEL_PROVIDER")
    if embedding_model_provider is None:
        raise ValueError("EMBEDDING_MODEL_PROVIDER environment variable must be set")

    bon_calculadora_toolgroup_id = os.environ.get("BON_CALCULADORA_TOOLGROUP_ID", None)
    if not bon_calculadora_toolgroup_id:
        raise ValueError("BON_CALCULADORA_TOOLGROUP_ID environment variable must be set")

    bon_calculadora_mcp_endpoint = os.environ.get("BON_CALCULADORA_MCP_ENDPOINT")
    if not bon_calculadora_mcp_endpoint:
        raise ValueError("BON_CALCULADORA_MCP_ENDPOINT environment variable must be set")

    host = os.environ.get("LLAMA_STACK_HOST")
    if not host:
        raise ValueError("LLAMA_STACK_HOST environment variable must be set")
    port = os.environ.get("LLAMA_STACK_PORT")
    if not port:
        raise ValueError("LLAMA_STACK_PORT environment variable must be set")
    secure = os.environ.get("LLAMA_STACK_SECURE", "false").lower() in ["true", "1", "yes"]
    

    # Initialize client
    client: LlamaStackClient = create_client(host=host, port=int(port), secure=secure)
    print(f"Connected to LlamaStack at {host}:{port}")

    # Register MCP Servers
    # mcp_toolgroup_id: str = register_mcp_toolgroup(client, bon_calculadora_toolgroup_id, bon_calculadora_mcp_endpoint)
    mcp_toolgroup_id: str = bon_calculadora_toolgroup_id  # Use a fixed ID for simplicity
    
    # Get models
    # model_id: str
    # embedding_model: object
    # model_id, embedding_model = get_models(client)
    
    # Register vector database
    embedding_model = get_embedding_model(client, embedding_model_id, embedding_model_provider)
    if not embedding_model:
        raise ValueError(f"Embedding model {embedding_model_id} not found for provider {embedding_model_provider}")
    print(f"Using embedding model: {embedding_model.identifier} (dimension: {embedding_model.metadata['embedding_dimension']})")
    vector_db_id: str = register_vector_db(client, embedding_model)
    
    # Load documents from folder
    docs_folder: str = os.environ.get("DOCS_FOLDER", "./docs")
    documents: List[RAGDocument] = load_documents_from_folder(docs_folder)
    
    # Insert documents into vector database
    if not args.no_insert:
        insert_documents(client, documents, vector_db_id, chunk_size_in_tokens=512)
    
    # Create system prompt for the agent
    sys_prompt: str = """
        You are a helpful assistant that can use tools to answer questions.
    """
    
    # If mode is 'auto', we will use the auto tool choice
    rag_tool = {
        "name": "builtin::rag/knowledge_search",
        "args": {
            "vector_db_ids": [vector_db_id],
        },
    }
    if args.mode == "auto":
        tools: List[Union[str, Any]] = [bon_calculadora_toolgroup_id, rag_tool]
    elif args.mode == "rag":
        tools: List[Union[str, Any]] = [rag_tool]
    elif args.mode == "mcp":
        tools: List[Union[str, Any]] = [bon_calculadora_toolgroup_id]
    else:
        tools: List[Union[str, Any]] = []
    print(f"Using tools: {tools}")

    # Create RAG agent with MCP toolgroup
    rag_agent: Agent = create_rag_agent_with_mcp(
        client, 
        model_id, 
        tools,
        sys_prompt
    )
    
    # Process user prompts
    user_prompts: List[str] = [
        # "Soy un padre soltero con 5 hijos de menos de 3 años, tengo derecho a la ayuda por excedencia en Navarra?",
        # "Soy un padre soltero con 5 hijos de menos de 8 años, tengo derecho a la ayuda por excedencia en Navarra?",
        # "Soy un padre soltero con 4 hijos, tengo derecho a la ayuda por excedencia en Navarra?",
        "Mi padre se ha roto la cadera y está hospitalizado, ¿tengo derecho a una ayuda por excedencia por cuidado de familiar en Navarra?",
        # "Mi padre se ha roto la cadera y está hospitalizado y tengo que estar con él permanentemente en el hospital, ¿tengo derecho a una ayuda por excedencia por cuidado de familiar en Navarra?",
        # "Acabamos de tener un hijo y tenemos otros 2 hijos de 2 y 5 años, ¿tengo derecho a una ayuda por excedencia por cuidado de hijos en Navarra?",
        # "La mujer de mi hermano está embarazada y vive en Navarra, ¿tiene ella derecho a una ayuda por excedencia por cuidado de hijos?",
        # "Mi cuñada está embarazada y vive en Navarra, ¿tiene derecho ella a una ayuda por excedencia por cuidado de hijos?",
    ]
    
    process_user_prompts(rag_agent, user_prompts)


if __name__ == "__main__":
    main()