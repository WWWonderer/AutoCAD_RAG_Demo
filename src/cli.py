from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_TOP_K, INDEX_PATH
from .rag import answer_question
from .retrieve import load_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions over an indexed PDF.")
    parser.add_argument("question", help='Question to ask, e.g. "How do I set drawing units?"')
    parser.add_argument("--index", default=str(INDEX_PATH), help="Path to previously built index.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="How many chunks to retrieve.")
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print retrieved source chunks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = Path(args.index)
    if not index_path.exists():
        raise SystemExit(
            f"Index not found at {index_path}. Run: python -m src.ingest --pdf <path-to-pdf>"
        )

    index = load_index(index_path)
    try:
        answer, sources = answer_question(index, args.question, top_k=args.top_k)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(answer)

    if args.show_sources and sources:
        print("\nSources:")
        for source in sources:
            page = (
                f"{source['page_start']}"
                if source["page_start"] == source["page_end"]
                else f"{source['page_start']}-{source['page_end']}"
            )
            heading = source["heading"] or "Untitled"
            print(
                f"[S{source['source_id']}] page {page}, chunk {source['chunk_id']}, "
                f"score={source['score']}"
            )
            print(f"    heading: {heading}")
            print(f"    {source['text'][:260].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
