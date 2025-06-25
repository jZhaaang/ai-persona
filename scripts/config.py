from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
SCRIPTS_DIR = PROJECT_DIR / "scripts"

RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
EMBED_DIR = DATA_DIR / "embed"
AUTHORS_PATH = DATA_DIR / "authors_map.json"
ENV_PATH = PROJECT_DIR / ".env"
