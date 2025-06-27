from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
SCRIPTS_DIR = PROJECT_DIR / "scripts"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed" / "json"
BATCH_DIR = DATA_DIR / "processed" / "batched"
CHUNKS_DIR = DATA_DIR / "chunks"
VECTORS_DIR = DATA_DIR / "vectors"
AUTHORS_PATH = DATA_DIR / "authors_map.json"
ENV_PATH = PROJECT_DIR / ".env"

CHUNK_MODEL = "gpt-4o-mini"
MAX_BATCH_MSGS = 30000
CHUNK_BATCH_SIZE = 250
EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 50
UPLOAD_BATCH_SIZE = 50
