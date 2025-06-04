import httpx
from typing import AsyncGenerator, Optional

async def stream_openai_chat(
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    url: str,
    openai_key: Optional[str] = None
) -> AsyncGenerator[str, None]:
    headers = {
        "Content-Type": "application/json"
    }
    if openai_key:
        headers["Authorization"] = f"Bearer {openai_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": True
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield f"{line}\n"

async def complete_openai_chat(
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    url: str,
    openai_key: Optional[str] = None
) -> str:
    headers = {
        "Content-Type": "application/json"
    }
    if openai_key:
        headers["Authorization"] = f"Bearer {openai_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False
    }

    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
