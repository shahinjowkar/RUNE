from __future__ import annotations

from openai import AsyncOpenAI, OpenAI

from core.config import settings

_client = OpenAI(api_key=settings.openai_api_key)
_async_client = AsyncOpenAI(api_key=settings.openai_api_key)


def embed(texts: list[str]) -> list[list[float]]:
    resp = _client.embeddings.create(model=settings.embed_model, input=texts)
    return [d.embedding for d in resp.data]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def chat(messages: list[dict], model: str | None = None, **kwargs) -> str:
    resp = _client.chat.completions.create(
        model=model or settings.chat_model,
        messages=messages,
        **kwargs,
    )
    return resp.choices[0].message.content


async def stream_chat(messages: list[dict], model: str | None = None):
    """Async generator that yields tokens from a streaming OpenAI response."""
    stream = await _async_client.chat.completions.create(
        model=model or settings.chat_model,
        messages=messages,
        stream=True,
    )
    async for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token
