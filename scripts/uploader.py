import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from config import ENV_PATH, VECTORS_DIR, UPLOAD_BATCH_SIZE
from utils import load_json_data

load_dotenv(dotenv_path=ENV_PATH)


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


def main():
    index = initialize_pinecone()
    print(f"Uploading vector files from {VECTORS_DIR}")

    for file in sorted(VECTORS_DIR.glob("*_vectors*.json")):
        vectors = load_json_data(file)

        print(f"Uploading {len(vectors)} vectors from {file}")
        for i in range(0, len(vectors), UPLOAD_BATCH_SIZE):
            batch = vectors[i : i + UPLOAD_BATCH_SIZE]
            index.upsert(batch)
            print(f"Uploaded {len(batch)} vectors from batch {i}-{i + len(batch)}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nScript finished in {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
