import openai
import os
import tiktoken
import time
from dotenv import load_dotenv
from config import ENV_PATH, CHUNKS_DIR, VECTORS_DIR, EMBED_MODEL, EMBED_BATCH_SIZE
from utils import load_json_data, write_json_data

VECTORS_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(dotenv_path=ENV_PATH)
openai.api_key = os.getenv("OPENAI_API_KEY")
tokenizer = tiktoken.get_encoding("cl100k_base")


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


def create_vectors(chunks):
    vectors = []
    print(f"Creating vectors for {len(chunks)} chunks")

    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        msgs = [
            f"Keywords: {chunk['keywords']}\n"
            + "\n".join(
                f"{msg['author_name']}: {msg['content']}" for msg in chunk["messages"]
            )
            for chunk in batch
        ]

        print(f"Embedding batch {i}-{i + len(batch)}")
        embeddings = embed_messages(msgs)

        for chunk, embedding, msg in zip(batch, embeddings, msgs):
            vectors.append(
                {
                    "id": chunk["chunk_id"],
                    "values": embedding,
                    "metadata": {
                        "keywords": chunk["keywords"],
                        "author_names": chunk["author_names"],
                        "text": msg,
                    },
                }
            )

    return vectors


def main():
    # embed every conversation chunk file
    for file_path in CHUNKS_DIR.glob("*_chunked*.json"):
        print(f"Processing chunk file: {file_path}")
        chunks = load_json_data(file_path)

        output_file = file_path.replace("_chunked", "_vectors")
        output_path = VECTORS_DIR / f"{output_file}"
        if output_path.exists():
            print(f"Output already exists for {file_path.name}, skipping")
            continue

        # create vectors with embeddings
        vectors = create_vectors(chunks)

        write_json_data(output_path, vectors)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(
            f"Saved chunk {file_path} with {len(vectors)} vectors to {output_path} ({size_mb:.2f} MB)"
        )


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nScript finished in {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
