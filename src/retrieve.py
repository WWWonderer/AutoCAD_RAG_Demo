from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

STOP_WORDS = set("a an and are as at be by for from how in is it of on or that the to what when where which with".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP_WORDS]


def build_index(chunks: list[dict], metadata: dict | None = None, k1: float = 1.5, b: float = 0.75) -> dict:
    postings: dict[str, dict[int, int]] = defaultdict(dict)
    doc_len: list[int] = []
    for chunk in chunks:
        doc_id = chunk["id"]
        tf = Counter(tokenize(chunk["text"]))
        doc_len.append(sum(tf.values()))
        for term, count in tf.items():
            postings[term][doc_id] = count
    packed = {term: [[doc_id, tf] for doc_id, tf in docs.items()] for term, docs in postings.items()}
    avg_len = sum(doc_len) / max(1, len(doc_len))
    return {
        "chunks": chunks,
        "postings": packed,
        "doc_len": doc_len,
        "avg_len": avg_len,
        "metadata": metadata or {},
        "k1": k1,
        "b": b,
    }


def save_index(index: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=True), encoding="utf-8")


def load_index(path: Path) -> dict:
    index = json.loads(path.read_text(encoding="utf-8"))
    index["avg_len"] = index.get("avg_len", sum(index["doc_len"]) / max(1, len(index["doc_len"])))
    index["k1"] = index.get("k1", 1.5)
    index["b"] = index.get("b", 0.75)
    return index


def _search(index: dict, query: str, top_k: int) -> list[dict]:
    terms = set(tokenize(query))
    if not terms or not index["chunks"]:
        return []
    scores: dict[int, float] = defaultdict(float)
    n_docs = len(index["chunks"])
    for term in terms:
        docs = index["postings"].get(term)
        if not docs:
            continue
        df = len(docs)
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        for doc_id, tf in docs:
            dl = index["doc_len"][doc_id] if doc_id < len(index["doc_len"]) else 0
            norm = index["k1"] * (1 - index["b"] + index["b"] * dl / max(1e-9, index["avg_len"]))
            scores[doc_id] += idf * (tf * (index["k1"] + 1)) / (tf + norm)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"chunk_id": i, "score": s, "chunk": index["chunks"][i]} for i, s in ranked[:top_k]]


def retrieve_chunks(index: dict, question: str, top_k: int = 6) -> list[dict]:
    hits = _search(index, question, top_k=top_k * 2)
    if not hits:
        return []

    score_floor = hits[0]["score"] * 0.25
    selected: list[dict] = []
    seen_prefixes: set[str] = set()
    for hit in hits:
        if hit["score"] < score_floor:
            continue
        prefix = hit["chunk"]["text"][:180].lower()
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        selected.append(
            {
                "source_id": len(selected) + 1,
                "chunk_id": hit["chunk_id"],
                "score": round(hit["score"], 3),
                "page_start": hit["chunk"]["page_start"],
                "page_end": hit["chunk"]["page_end"],
                "heading": hit["chunk"].get("heading", ""),
                "text": hit["chunk"]["text"],
            }
        )
        if len(selected) == top_k:
            break
    return selected
