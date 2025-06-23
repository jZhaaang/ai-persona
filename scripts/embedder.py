import openai
import json
import os
import tiktoken
import time
from dotenv import load_dotenv
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

openai.api_key = os.getenv("OPENAI_API_KEY")
tokenizer = tiktoken.get_encoding("cl100k_base")

EMBED_MODEL = "text-embedding-3-small"
SCRIPTS_DIR = Path(__file__).parent
DATA_DIR = (SCRIPTS_DIR / ".." / "data").resolve()
CHUNKS_DIR = DATA_DIR / "clean" / "chunks"
OUTPUT_DIR = os.path.join(DATA_DIR, "embed")
BATCH_SIZE = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_json_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_tokens(texts):
    return sum(len(tokenizer.encode(text)) for text in texts)


def embed_messages(msgs, retries=3):
    try:
        response = openai.embeddings.create(input=msgs, model=EMBED_MODEL)
        return [res.embedding for res in response.data]
    except openai.OpenAIError as e:
        print("API Error when creating embed: ", e)
        if retries > 0:
            time.sleep(1)
            return embed_messages(msgs, retries - 1)
        else:
            raise


if __name__ == "__main__":
    start_time = time.time()
    authors_file = os.path.join(DATA_DIR, "authors_map.json")
    authors = load_json_data(authors_file)

    chunk_files = sorted(CHUNKS_DIR.glob("chunked_batch_*.json"))
    total_chunks = 0
    total_token_count = 0

    for file in chunk_files:
        print(f"Processing batch file: {file}")
        batch_name = file.stem.replace("chunked_", "")
        embed_path = Path(OUTPUT_DIR) / f"embedded_{batch_name}.json"

        if embed_path.exists():
            print(f"Embed batch {batch_name} already exists. Skipping.")
            continue

        chunks = load_json_data(file)
        vectors = []

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            msgs = [
                f"Topic: {chunk['topic']}\n"
                + "\n".join(
                    f"{msg['author_name']}: {msg['content']}"
                    for msg in chunk["messages"]
                )
                for chunk in batch
            ]

            token_count = count_tokens(msgs)
            total_token_count += token_count

            print(f"Embedding batch {i}-{i + len(batch)}, ({token_count} tokens)")
            embeddings = embed_messages(msgs)

            for chunk, embedding, msg in zip(batch, embeddings, msgs):
                vectors.append(
                    {
                        "id": chunk["chunk_id"],
                        "values": embedding,
                        "metadata": {
                            "topic": chunk["topic"],
                            "author_names": chunk["author_names"],
                            "text": msg,
                        },
                    }
                )

        total_chunks += len(chunks)
        with embed_path.open("w", encoding="utf-8") as f:
            json.dump(vectors, f, indent=2, ensure_ascii=False)

        size_mb = os.path.getsize(embed_path) / (1024 * 1024)
        print(
            f"Saved batch {batch_name} with {len(vectors)} vectors to {embed_path} ({size_mb:.2f} MB)"
        )

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nScript finished in {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
    print(
        f"Finished embedding {total_chunks} chunks, total of {total_token_count} tokens"
    )
