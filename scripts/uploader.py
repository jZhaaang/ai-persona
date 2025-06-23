import json
import os
import time
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

SCRIPTS_DIR = Path(__file__).parent
DATA_DIR = (SCRIPTS_DIR / ".." / "data").resolve()
EMBED_DIR = DATA_DIR / "embed"
BATCH_SIZE = 50


def load_json_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def initialize_pinecone():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX")

    pc = Pinecone(api_key=pinecone_api_key)

    if pinecone_index_name not in pc.list_indexes().names():
        print(f"Creating index '{pinecone_index_name}'...")
        pc.create_index(
            name=pinecone_index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.Index(pinecone_index_name)


if __name__ == "__main__":
    start_time = time.time()
    index = initialize_pinecone()
    print(f"Uploading embedded files from {EMBED_DIR}")

    for file in sorted(EMBED_DIR.glob("embedded_batch_*.json")):
        vectors = load_json_data(file)

        print(f"Uploading {len(vectors)} vectors from {file}")
        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i : i + BATCH_SIZE]
            index.upsert(batch)
            print(f"Uploaded {len(batch)} vectors from batch {i}-{i + len(batch)}")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nScript finished in {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
