from __future__ import annotations

import os
from pathlib import Path

ARTIFACT_DIR = Path("artifacts")
INDEX_PATH = ARTIFACT_DIR / "index.json"

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 140
DEFAULT_TOP_K = 6

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_TEMPERATURE = 0.1
