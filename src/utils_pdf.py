from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
PAGE_NUM_RE = re.compile(r"(page\s+)?\d+(\s+of\s+\d+)?", re.I)
SECTION_RE = re.compile(r"^(\d+(\.\d+)*\s+)?[A-Z][\w /()-]{2,}$")


def _require_pypdf():
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'pypdf'. Install with: pip install -r requirements.txt"
        ) from exc
    return PdfReader


def _clean(line: str) -> str:
    return re.sub(r"\s+", " ", line.replace("\xa0", " ").strip())


def _norm(line: str) -> str:
    return re.sub(r"\d+", "#", line.lower()).strip()


def _is_heading(line: str) -> bool:
    return 4 <= len(line) <= 90 and (line.endswith(":") or bool(SECTION_RE.match(line)) or line.isupper())


def _is_noise(line: str, repeated: set[str]) -> bool:
    return not line or len(line) <= 2 or bool(PAGE_NUM_RE.fullmatch(line)) or (len(line) <= 70 and _norm(line) in repeated)


def _split(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    parts = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()] or [text.strip()]
    chunks, current = [], ""
    for part in parts:
        candidate = f"{current} {part}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = f"{current[-overlap_chars:].strip()} {part}".strip() if overlap_chars > 0 else part
        else:
            current = candidate
    if current:
        chunks.append(current)
    if all(len(c) <= max_chars for c in chunks):
        return chunks
    step = max(1, max_chars - overlap_chars)
    return [c[i : i + max_chars].strip() for c in chunks for i in range(0, len(c), step) if c[i : i + max_chars].strip()]


def extract_chunks(pdf_path: Path, max_chars: int, overlap_chars: int) -> tuple[list[dict], dict]:
    reader = _require_pypdf()(str(pdf_path))
    pages = [(i, [_clean(x) for x in ((p.extract_text() or "").replace("\r", "\n")).splitlines()]) for i, p in enumerate(reader.pages, start=1)]
    counts = Counter(_norm(line) for _, lines in pages for line in lines if line and len(line) <= 70)
    repeated = {line for line, n in counts.items() if n >= max(4, len(pages) // 5)}

    blocks, heading = [], ""
    for page_no, lines in pages:
        para: list[str] = []
        for line in lines:
            if _is_heading(line):
                if para:
                    blocks.append({"heading": heading, "text": " ".join(para), "page_start": page_no, "page_end": page_no})
                    para = []
                heading = line
                continue
            if _is_noise(line, repeated):
                if para:
                    blocks.append({"heading": heading, "text": " ".join(para), "page_start": page_no, "page_end": page_no})
                    para = []
                continue
            para.append(line)
        if para:
            blocks.append({"heading": heading, "text": " ".join(para), "page_start": page_no, "page_end": page_no})

    merged: list[dict] = []
    for block in blocks:
        text = re.sub(r"(?<=\w)- (?=\w)", "", block["text"]).strip()
        if not text:
            continue
        if merged and merged[-1]["heading"] == block["heading"] and block["page_start"] <= merged[-1]["page_end"] + 1:
            merged[-1]["text"] = f"{merged[-1]['text']}\n\n{text}".strip()
            merged[-1]["page_end"] = block["page_end"]
        else:
            merged.append({**block, "text": text})

    chunks: list[dict] = []
    for block in merged:
        for part in _split(block["text"], max_chars=max_chars, overlap_chars=overlap_chars):
            if len(part) < 80:
                continue
            chunks.append(
                {
                    "id": len(chunks),
                    "text": f"{block['heading']}\n{part}".strip() if block["heading"] else part,
                    "heading": block["heading"],
                    "page_start": block["page_start"],
                    "page_end": block["page_end"],
                }
            )

    return chunks, {"pages_total": len(pages), "blocks_total": len(merged), "chunks_total": len(chunks)}
