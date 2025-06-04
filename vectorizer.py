import httpx
from typing import Optional

def prefix_tunning(text: str, model: str) -> str:
    # If model contains "multilingual-e5", prefix the text with 'query: '
    if "multilingual-e5" in model.lower():
        return f"query: {text}"
    # If model contains "bge-e", prefix the text with 'query: '
    elif "bge-e" in model.lower():
        return f"query: {text}"
    # If model contains "gte-", prefix the text with 'query: '
    elif "bge-e" in model.lower():
        return f"query: {text}"
    return text

async def get_vector(text: str, url: str, model: str, openai_key: Optional[str] = None) -> list[float]:
    headers = {
        "Content-Type": "application/json"
    }
    if openai_key:
        headers["Authorization"] = f"Bearer {openai_key}"
    payload = {
        "input": prefix_tunning(text, model),
        "model": model
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]