import json
import uuid
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import tiktoken


def load_json_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


SCRIPTS_DIR = Path(__file__).parent
DATA_DIR = (SCRIPTS_DIR / ".." / "data").resolve()
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
CHUNKS_DIR = os.path.join(CLEAN_DIR, "chunks")
AUTHORS_PATH = os.path.join(DATA_DIR, "authors_map.json")
MESSAGES_PATH = os.path.join(CLEAN_DIR, "filtered_messages.json")
MAX_TOKENS = 300
TIME_THRESHOLD = 5 * 60
SUMMARY_THRESHOLD = 15
BATCH_SIZE = 2000

AUTHORS_MAP = load_json_data(AUTHORS_PATH)
MESSAGES = load_json_data(MESSAGES_PATH)
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
tokenizer = tiktoken.get_encoding("cl100k_base")


def num_tokens(text):
    return len(tokenizer.encode(text))


def parse_timestamp(timestamp):
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def get_chunk_id():
    return str(uuid.uuid4())


def summarize_chunk(msgs):
    messages = "\n".join([f"{msg['author_name']}: {msg['content']}" for msg in msgs])

    prompt = (
        "Summarize this conversation into concise description (1-2 sentences), focusing on the main topic and who's speaking. Try to describe in the tone and vocabulary of the users messaging:\n\n:"
        + f"{messages}"
    )

    try:
        start_time = time.time()
        response = client.responses.create(model="gpt-4.1-mini", input=prompt)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"({msgs[0]['message_id']}) ({elapsed:.2f}s) - {response.output_text}")

        return response.output_text
    except Exception as e:
        print("Error summarizing chunk: ", e)
        return "Unknown topic"


def chunk_messages(msgs):
    chunks = []
    current_chunk = []
    current_tokens = 0
    current_authors = set()
    last_timestamp = None

    time_splits = 0
    token_splits = 0

    for msg in msgs:
        content_tokens = num_tokens(msg["content"])
        timestamp = parse_timestamp(msg["timestamp"])

        time_gap = (timestamp - last_timestamp).total_seconds() if last_timestamp else 0
        split_by_time = time_gap > TIME_THRESHOLD
        split_by_tokens = current_tokens + content_tokens > MAX_TOKENS

        if split_by_time or split_by_tokens:
            time_splits += 1 if split_by_time else 0
            token_splits += 1 if split_by_tokens else 0
            if current_chunk:
                chunks.append(
                    {
                        "chunk_id": get_chunk_id(),
                        "messages": current_chunk,
                        "author_names": list(current_authors),
                        "start_timestamp": current_chunk[0]["timestamp"],
                        "end_timestamp": current_chunk[-1]["timestamp"],
                    }
                )

            current_chunk = []
            current_authors = set()
            current_tokens = 0

        author_name = AUTHORS_MAP.get(msg["author_id"], msg["author_id"])
        current_chunk.append({**msg, "author_name": author_name})
        current_authors.add(author_name)
        current_tokens += content_tokens
        last_timestamp = timestamp

    if current_chunk:
        chunks.append(
            {
                "chunk_id": get_chunk_id(),
                "messages": current_chunk,
                "author_names": list(current_authors),
                "start_timestamp": current_chunk[0]["timestamp"],
                "end_timestamp": current_chunk[-1]["timestamp"],
            }
        )

    print(
        f"Finished chunking messages into {len(chunks)} chunks\n{time_splits} conversation chunks, {token_splits} chunks that exceeded {MAX_TOKENS} tokens."
    )

    return chunks


def split_chunks(chunks, batch_size):
    for i in range(0, len(chunks), batch_size):
        yield chunks[i : i + batch_size], i // batch_size + 1


if __name__ == "__main__":
    start_time = time.time()
    total_tokens = 0
    summary_count = 0
    output_path = os.path.join(CLEAN_DIR, "chunked_messages.json")
    print(f"Processing file: {MESSAGES_PATH}")

    all_chunks = chunk_messages(MESSAGES)

    for chunk_batch, batch_num in split_chunks(all_chunks, BATCH_SIZE):
        batch_path = Path(CHUNKS_DIR) / f"chunked_batch_{batch_num:03}.json"

        if batch_path.exists():
            print(f"Batch {batch_num:03} already exists. Skipping.")
            continue

        finalized = []
        for chunk in chunk_batch:
            if len(chunk["messages"]) > SUMMARY_THRESHOLD:
                summary_count += 1
                topic = summarize_chunk(chunk["messages"])
            else:
                topic = "Short exchange"
            chunk["topic"] = topic
            finalized.append(chunk)

        batch_path.parent.mkdir(parents=True, exist_ok=True)

        with batch_path.open("w", encoding="utf-8") as f:
            json.dump(finalized, f, indent=2, ensure_ascii=False)

        size_mb = os.path.getsize(batch_path) / (1024 * 1024)
        print(
            f"Saved batch {batch_num:03} with {len(finalized)} chunks and {summary_count} summaries to {batch_path} ({size_mb:.2f} MB)"
        )

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nScript finished in {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
