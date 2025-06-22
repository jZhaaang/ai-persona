import openai
import json
import os
import tiktoken
from time import sleep
from dotenv import load_dotenv
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

openai.api_key = os.getenv("OPENAI_API_KEY")
tokenizer = tiktoken.get_encoding("cl100k_base")

EMBED_MODEL = "text-embedding-3-small"
SCRIPTS_DIR = Path(__file__).parent
DATA_DIR = (SCRIPTS_DIR / ".." / "data").resolve()
CHUNKS_DIR = os.path.join(DATA_DIR, "clean")
OUTPUT_DIR = os.path.join(DATA_DIR, "embed")
BATCH_SIZE = 25

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
            sleep(1)
            return embed_messages(msgs, retries - 1)
        else:
            raise


if __name__ == "__main__":
    chunks_file = os.path.join(CHUNKS_DIR, "chunked_messages.json")
    authors_file = os.path.join(DATA_DIR, "authors_map.json")
    chunks = load_json_data(chunks_file)
    authors = load_json_data(authors_file)
    embeddings = []
    total_token_count = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        msgs = [
            "\n".join(
                [
                    f"{authors.get(msg.get('author_id'), 'Unknown')}: {msg['content']}"
                    for msg in chunk["messages"]
                ]
            )
            for chunk in batch
        ]
        token_count = count_tokens(msgs)
        print(f"Embedding batch {i}-{i + len(batch)}, {token_count} tokens")
        total_token_count += token_count

        batch_embeddings = embed_messages(msgs)

        for chunk, embedding, msg in zip(batch, batch_embeddings, msgs):
            embeddings.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "embedding": embedding,
                    "text": msg,
                }
            )

    output_path = os.path.join(OUTPUT_DIR, "embedded_messages.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, indent=2, ensure_ascii=False)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved {len(chunks)} embeds to {output_path} ({size_mb:.2f} MB)")
    print(f"Total token count {total_token_count}")
