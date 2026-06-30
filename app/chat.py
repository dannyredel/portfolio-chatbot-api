import json
from collections.abc import AsyncGenerator

from anthropic import Anthropic

from app.config import settings
from app.embeddings import embed
from app.prompts import SYSTEM_PROMPT, build_user_message
from app.retrieval import match_documents

_anthropic = Anthropic(api_key=settings.anthropic_api_key)


def _retrieve(question: str) -> list[dict]:
    q_emb = embed(question)
    return match_documents(q_emb, top_k=settings.top_k)


def answer_stream(question: str) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted chunks: {type: 'sources'|'token'|'done', ...}."""
    chunks = _retrieve(question)
    user_msg = build_user_message(question, chunks)

    sources = [
        {
            "source": (c.get("metadata") or {}).get("source"),
            "section": (c.get("metadata") or {}).get("section"),
            "similarity": c.get("similarity"),
        }
        for c in chunks
    ]

    async def gen() -> AsyncGenerator[str, None]:
        yield _sse({"type": "sources", "sources": sources})
        with _anthropic.messages.stream(
            model=settings.anthropic_model,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            for text in stream.text_stream:
                yield _sse({"type": "token", "text": text})
        yield _sse({"type": "done"})

    return gen()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
