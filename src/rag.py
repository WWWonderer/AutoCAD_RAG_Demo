from __future__ import annotations

from .config import DEFAULT_TOP_K, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE
from .retrieve import retrieve_chunks


def _require_openai():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'openai'. Install with: pip install -r requirements.txt"
        ) from exc
    return OpenAI


def _format_context(sources: list[dict]) -> str:
    parts = []
    for source in sources:
        pages = f"{source['page_start']}" if source["page_start"] == source["page_end"] else f"{source['page_start']}-{source['page_end']}"
        heading = source["heading"] or "Untitled"
        parts.append(
            f"[S{source['source_id']}] page {pages} | {heading}\n{source['text']}"
        )
    return "\n\n".join(parts)


def answer_question(index: dict, question: str, top_k: int = DEFAULT_TOP_K) -> tuple[str, list[dict]]:
    sources = retrieve_chunks(index, question, top_k=top_k)
    if not sources:
        return "I could not find relevant content in the indexed PDF.", []

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running the CLI.")

    client = _require_openai()(api_key=OPENAI_API_KEY)
    prompt = (
        "Answer using only the provided sources.\n"
        "Add citations inline in the form [S1], [S2].\n"
        "If the sources are insufficient, say so clearly.\n\n"
        f"Question:\n{question}\n\n"
        f"Sources:\n{_format_context(sources)}"
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        messages=[
            {"role": "system", "content": "You are a precise technical assistant for AutoCAD documentation."},
            {"role": "user", "content": prompt},
        ],
    )
    answer = (response.choices[0].message.content or "").strip()
    return (answer or "I could not generate an answer from the provided sources."), sources
