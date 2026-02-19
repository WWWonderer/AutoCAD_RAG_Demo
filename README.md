# AutoCAD PDF RAG Demo

Small, readable RAG prototype for a local PDF manual. Can be run on CPU.

## First step: add the PDF

1. Download the sample PDF from:
   `https://static.sdcpublications.com/pdfsample/978-1-63057-655-4-1-3ltgcdetqz.pdf`
2. Save it in `data/` using a suitable filename such as: 
   `autocad_2025_tutorial.pdf`

## What it supports

1. Ingest and chunk a PDF into a searchable index:

```bash
python -m src.ingest --pdf data/autocad_2025_tutorial.pdf
```

2. Ask questions with LLM-grounded citations:

```bash
python -m src.cli "Is clockwise positive or negative in AutoCAD?" --show-sources
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your_api_key"
```

## Design

- `src/config.py`: shared settings.
- `src/utils_pdf.py`: PDF parsing, noise filtering, section-aware chunking.
- `src/retrieve.py`: BM25 indexing + chunk retrieval.
- `src/rag.py`: OpenAI call with grounded prompt and inline `[S#]` citations.
- `src/ingest.py`: chunk + index build into `artifacts/index.json`.
- `src/cli.py`: command-line query interface.

## Useful options

- `--chunk-size` and `--chunk-overlap` in `src.ingest` tune retrieval quality.
- `--top-k` in `src.cli` adjusts how many chunks are considered for the answer.

## Notes

- The current implementation prioritizes readability and simplicity.
- Future improvements include semantic-aware chunking, embedding based indexing (instead of BM25), table extraction, and OCR for images.
