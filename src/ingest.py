from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, INDEX_PATH
from .retrieve import build_index, save_index
from .utils_pdf import extract_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a lightweight BM25 RAG index from a PDF.")
    parser.add_argument("--pdf", required=True, help="Path to PDF file.")
    parser.add_argument("--index", default=str(INDEX_PATH), help="Where to write the index JSON.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Max chars per chunk.")
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Trailing chars carried into next chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    try:
        chunks, stats = extract_chunks(
            pdf_path,
            max_chars=args.chunk_size,
            overlap_chars=args.chunk_overlap,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    if not chunks:
        raise SystemExit("No text extracted from PDF. Try a text-based PDF (not scanned images).")

    metadata = {
        "pdf_path": str(pdf_path),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        **stats,
    }
    index = build_index(chunks, metadata=metadata)
    index_path = Path(args.index)
    save_index(index, index_path)

    print(f"Indexed: {pdf_path}")
    print(f"Pages scanned: {stats['pages_total']}")
    print(f"Semantic blocks: {stats['blocks_total']}")
    print(f"Chunks: {stats['chunks_total']}")
    print(f"Index written to: {index_path}")


if __name__ == "__main__":
    main()
