import json
import uuid
import os
from datetime import datetime
from pathlib import Path
import tiktoken

SCRIPTS_DIR = Path(__file__).parent
DATA_DIR = (SCRIPTS_DIR / ".." / "data" / "clean").resolve()
MAX_TOKENS = 300
TIME_THRESHOLD = 5 * 60

tokenizer = tiktoken.get_encoding("cl100k_base")


def load_json_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def num_tokens(text):
    return len(tokenizer.encode(text))


def parse_timestamp(timestamp):
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def get_chunk_id():
    return str(uuid.uuid4())


def chunk_messages(msgs):
    chunks = []
    current_chunk = []
    current_tokens = 0
    current_authors = set()
    last_timestamp = None

    time_splits = 0
    token_splits = 0

    def push_chunk():
        nonlocal current_chunk, current_authors, current_tokens
        if not current_chunk:
            return None
        cid = get_chunk_id()
        chunks.append(
            {
                "chunk_id": cid,
                "messages": current_chunk,
                "author_ids": list(current_authors),
                "start_timestamp": current_chunk[0]["timestamp"],
                "end_timestamp": current_chunk[-1]["timestamp"],
                "ref_chunk_id": last_chunk_id,
            }
        )
        current_chunk = []
        current_authors = set()
        current_tokens = 0

        return cid

    last_chunk_id = None

    for msg in msgs:
        content_tokens = num_tokens(msg["content"])
        timestamp = parse_timestamp(msg["timestamp"])

        time_gap = (timestamp - last_timestamp).total_seconds() if last_timestamp else 0
        split_by_time = time_gap > TIME_THRESHOLD
        split_by_tokens = current_tokens + content_tokens > MAX_TOKENS

        if split_by_time or split_by_tokens:
            time_splits += 1 if split_by_time else 0
            token_splits += 1 if split_by_tokens else 0
            new_chunk_id = push_chunk()
            last_chunk_id = None if split_by_time else new_chunk_id

        current_chunk.append(msg)
        current_authors.add(msg["author_id"])
        current_tokens += content_tokens
        last_timestamp = timestamp

    new_chunk_id = push_chunk()
    if last_chunk_id is not None and new_chunk_id is not None:
        chunks[-1]["ref_chunk_id"] = last_chunk_id

    print(
        f"Finished chunking messages into {len(chunks)} chunks\n{time_splits} conversation chunks, {token_splits} chunks that exceeded {MAX_TOKENS} tokens."
    )

    return chunks


if __name__ == "__main__":
    input_path = os.path.join(DATA_DIR, "filtered_messages.json")
    output_path = os.path.join(DATA_DIR, "chunked_messages.json")
    msgs = load_json_data(input_path)
    print(f"Processing file: {input_path}")

    chunks = chunk_messages(msgs)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved {len(chunks)} chunks to {output_path} ({size_mb:.2f} MB)")
    average_msgs_per_chunk = len(msgs) / len(chunks)
    print(f"Average {average_msgs_per_chunk} messages per chunk.")
