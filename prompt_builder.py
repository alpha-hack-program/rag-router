import os
from regex import P
import yaml

DEFAULT_PROMPT_KEY = "default"
CONTEXT_PROMPT_KEY = "context"

PROMPTS_PATH: str = os.getenv("PROMPTS_PATH", "")
if not PROMPTS_PATH:
    raise ValueError("PROMPTS_PATH environment variable is not set.")

try:
    with open(PROMPTS_PATH, "r") as f:
        # Load the prompt configuration from the yaml file
        PROMPTS = yaml.safe_load(f)
        print(f"Loaded prompt configuration from {PROMPTS_PATH}")
        print(f"Available prompts: {PROMPTS}")
        # Get the prompt map from the loaded configuration
        PROMPT_LIST = PROMPTS.get("prompts", {})
        if not PROMPT_LIST:
            raise ValueError("No prompts found in the configuration file.")
        # Convert the list of prompts to a dictionary with keys as prompt names
        PROMPT_MAP = {prompt["name"]: prompt for prompt in PROMPT_LIST}
        print(f"Initial prompt map: {PROMPT_MAP}")
        # Ensure that the default and context prompts are present
        if DEFAULT_PROMPT_KEY not in PROMPT_MAP:
            raise ValueError(f"Default prompt '{DEFAULT_PROMPT_KEY}' not found in the configuration.")
        if CONTEXT_PROMPT_KEY not in PROMPT_MAP:
            raise ValueError(f"Context prompt '{CONTEXT_PROMPT_KEY}' not found in the configuration.")
        print(f"Final prompt map: {PROMPT_MAP}")
except Exception as e:
    raise RuntimeError(f"Failed to load prompt configuration from {PROMPTS_PATH}: {e}")

def build_prompt(query: str, docs: list[str]) -> str:
    if not PROMPT_MAP:
        raise ValueError("PROMPT_MAP is empty. Please check the configuration file.")
    else:
        print(f"Using prompt configuration: {PROMPT_MAP}")
    
    # If no documents are provided, return a simple prompt
    if not docs:
        # Load "default" prompt from the configuration: prompts->
        no_context_prompt = PROMPT_MAP.get(DEFAULT_PROMPT_KEY, {})
        template = no_context_prompt.get("template")
        if not template:
            raise ValueError(f"No 'template' found for '{DEFAULT_PROMPT_KEY}' prompt in configuration.")
        return template.format(query=query)
    # Join the documents with a separator
    context = "\n---\n".join(docs)
    # Because there is context use the 'context' prompt template
    general_prompt = PROMPT_MAP.get(CONTEXT_PROMPT_KEY, {})
    template = general_prompt.get("template")
    if not template:
        raise ValueError(f"No 'template' found for '{CONTEXT_PROMPT_KEY}' prompt in configuration.")
    return template.format(context=context, query=query)
