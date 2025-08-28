"""
Agent builder module for creating Llama Stack agents.

This module provides functions to build individual agents or multiple agents
from a list of model identifiers using the LlamaStackClient.
"""

from typing import Optional, List, Union, Any, Dict

from llama_stack_client import LlamaStackClient

from llama_stack_client.types.model import Model
from llama_stack_client.types.shared_params.document import Document as RAGDocument
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger as AgentEventLogger

def build_agent(
    client: LlamaStackClient,
    model_id: str,
    tools: Optional[List[Union[str, Any]]] = None,
    instructions: str = "You are a helpful assistant"
) -> Agent:
    """
    Creates and returns an Agent instance with the specified parameters.

    Args:
        client (LlamaStackClient): The LlamaStackClient instance for interacting 
            with the Llama Stack server.
        model_id (str): The identifier for the model to be used by the agent.
        tools (Optional[List[Union[str, Any]]], optional): A list of tools or 
            tool identifiers to be used by the agent. Can include strings for 
            tool names or tool objects. Defaults to None.
        instructions (str, optional): Instructions defining the agent's behavior 
            and personality. Defaults to "You are a helpful assistant".

    Returns:
        Agent: A configured Agent instance ready to handle conversations.

    Raises:
        ValueError: If the model_id is not valid or not available.
        ConnectionError: If the client cannot connect to the Llama Stack server.

    Example:
        >>> from llama_stack_client import LlamaStackClient
        >>> client = LlamaStackClient(base_url="http://localhost:8321")
        >>> agent = build_agent(
        ...     client=client,
        ...     model_id="llama-3-1-8b-instruct",
        ...     tools=["calculator", "web_search"],
        ...     instructions="You are a helpful research assistant."
        ... )
    """
    print(f"Creating agent with model: {model_id}")
    if not model_id:
        raise ValueError("Model ID is required to create the agent")
    if not tools:
        tools = []

    try:
        # Create the agent using the Llama Stack client
        agent_config = {
            "model": model_id,
            "instructions": instructions,
        }
        
        # Add tools if provided
        if tools:
            agent_config["tools"] = tools
            
        # Create the agent using the client's agents endpoint
        print(f"Using tools: {tools}")
        print(f"Agent instructions: {instructions}")
        agent = Agent(
            client,
            model=model_id,
            instructions=instructions,
            enable_session_persistence=False,
            tools=tools or [],
            tool_config={"tool_choice": "auto" if tools else "none"},
        )
        
        return agent
        
    except Exception as e:
        raise ValueError(f"Failed to create agent with model '{model_id}': {str(e)}")


def build_agents(
    client: LlamaStackClient,
    model_ids: List[str],
    tools: Optional[List[Union[str, Any]]] = None,
    instructions: str = "You are a helpful assistant"
) -> Dict[str, Agent]:
    """
    Creates and returns a dictionary of Agent instances for each model_id provided.

    Args:
        client (LlamaStackClient): The LlamaStackClient instance for interacting 
            with the Llama Stack server.
        model_ids (List[str]): A list of model identifiers for which agents 
            will be created.
        tools (Optional[List[Union[str, Any]]], optional): A list of tools or 
            tool identifiers to be used by each agent. Will be applied to all 
            agents. Defaults to None.
        instructions (str, optional): Instructions for each agent's behavior. 
            Will be applied to all agents. Defaults to "You are a helpful assistant".

    Returns:
        Dict[str, Agent]: A dictionary mapping each model_id to its corresponding 
            Agent instance.

    Raises:
        ValueError: If any of the model_ids are not valid or if the model_ids 
            list is empty.
        ConnectionError: If the client cannot connect to the Llama Stack server.

    Example:
        >>> from llama_stack_client import LlamaStackClient
        >>> client = LlamaStackClient(base_url="http://localhost:8321")
        >>> model_list = ["llama-3-1-8b-instruct", "granite-3-1-8b-instruct"]
        >>> agents = build_agents(
        ...     client=client,
        ...     model_ids=model_list,
        ...     tools=["calculator"],
        ...     instructions="You are a helpful assistant specializing in math."
        ... )
        >>> # Access specific agent
        >>> math_agent = agents["llama-3-1-8b-instruct"]
    """
    if not model_ids:
        raise ValueError("model_ids list cannot be empty")
    
    agents = {}
    failed_models = []
    
    for model_id in model_ids:
        try:
            agent = build_agent(
                client=client,
                model_id=model_id,
                tools=tools,
                instructions=instructions
            )
            agents[model_id] = agent
        except Exception as e:
            failed_models.append((model_id, str(e)))
            continue
    
    # If some models failed but others succeeded, log warning but return successful ones
    if failed_models:
        failed_list = [f"{model}: {error}" for model, error in failed_models]
        print(f"Warning: Failed to create agents for the following models: {failed_list}")
    
    # If no agents were created successfully, raise an error
    if not agents:
        raise ValueError(f"Failed to create agents for all provided models: {model_ids}")
    
    return agents


def get_available_models(client: LlamaStackClient) -> List[str]:
    """
    Helper function to get available model IDs from the Llama Stack server.

    Args:
        client (LlamaStackClient): The LlamaStackClient instance.

    Returns:
        List[str]: A list of available model identifiers.

    Raises:
        ConnectionError: If the client cannot connect to the Llama Stack server.
    """
    try:
        models = client.models.list()
        return [model.identifier for model in models if hasattr(model, 'model_type') and model.model_type == "llm"]
    except Exception as e:
        raise ConnectionError(f"Failed to retrieve available models: {str(e)}")

def build_agents_for_available_models(client: LlamaStackClient) -> Dict[str, Agent]:
    """
    Creates and returns a dictionary of Agent instances for each available model.
    """
    model_ids = get_available_models(client)
    return build_agents(client, model_ids)

def build_llama_stack_client(base_url: str) -> LlamaStackClient:
    """
    Creates and returns a LlamaStackClient instance.
    """
    return LlamaStackClient(base_url=base_url)