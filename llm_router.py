import json, httpx
from typing import AsyncGenerator, Optional

PREFIX = "data: "
DONE   = f"{PREFIX}[DONE]"

async def stream_openai_chat(
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    url: str,
    openai_key: Optional[str] = None,
    references: Optional[list[str]] = None,
) -> AsyncGenerator[str, None]:
    """
    Yield an SSE stream identical to the backend's, plus a well-formed
    references section and a closing chunk whose finish_reason equals the
    backend's last non-null value.
    """
    headers = {"Content-Type": "application/json"}
    if openai_key:
        headers["Authorization"] = f"Bearer {openai_key}"

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": True,
    }

    last_meta         : dict | None   = None   # id/object/created/model
    backend_fin_reason: str | None    = None   # e.g. "length" or "stop"
    has_stop_key      : bool | None   = None   # does backend include stop_reason?

    async with httpx.AsyncClient(timeout=None) as client, \
               client.stream("POST", url, headers=headers, json=body) as resp:
        resp.raise_for_status()

        async for raw in resp.aiter_lines():
            # Ignore keep-alives (": …") and delimiter blanks from backend.
            if not raw.startswith(PREFIX):
                continue
            if raw == DONE:
                break                              # backend finished

            payload = raw[len(PREFIX):]            # JSON without "data: "
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                # forward malformed chunk exactly as we got it
                yield raw + "\n\n"
                continue

            # Always remember meta fields from the *original* chunk
            last_meta = {k: obj[k] for k in ("id", "object", "created", "model")}
            choice    = obj["choices"][0]
            has_stop_key = "stop_reason" in choice

            if choice.get("finish_reason") is not None:
                # Backend's terminal chunk – forward a *modified* copy
                backend_fin_reason = choice["finish_reason"]

                # rewrite: finish_reason -> null, drop stop_reason
                choice["finish_reason"] = None
                choice.pop("stop_reason", None)

            # Forward (possibly modified) chunk
            yield f"{PREFIX}{json.dumps(obj, separators=(',',':'))}\n"

    # ---------------------------------------------------------------------
    if references and last_meta:
        def ref_chunk(text: str):
            choice = {
                "index": 0,
                "delta": {"content": text},
                "logprobs": None,
                "finish_reason": None,
            }
            return last_meta | {"choices": [choice]}

        # Header + each reference line
        newline = '\n'
        for line in ["\n**References:**", *references]:
            yield f"{PREFIX}{json.dumps(ref_chunk(newline), separators=(',',':'))}\n"
            yield f"{PREFIX}{json.dumps(ref_chunk(line), separators=(',',':'))}\n"

    # Closing synthetic chunk with the remembered finish_reason
    if last_meta and backend_fin_reason is not None:
        closing_choice = {
            "index": 0,
            "delta": {},
            "logprobs": None,
            "finish_reason": backend_fin_reason,
        }
        if has_stop_key:
            closing_choice["stop_reason"] = None
        yield f"{PREFIX}{json.dumps(last_meta | {'choices':[closing_choice]}, separators=(',',':'))}\n"

    yield f"{DONE}\n"           # single sentinel, no extras

async def complete_openai_chat(
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    url: str,
    openai_key: Optional[str] = None,
    references: Optional[list[str]] = None
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
        content = response.json()["choices"][0]["message"]["content"]

        # Append references if available as lines
        if references:
            content += "\n\n**References:**\n" + "\n".join(references)
        return content

