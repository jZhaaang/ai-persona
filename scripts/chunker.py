import json
import time
import openai
import re
from dotenv import load_dotenv
from datetime import datetime
from config import (
    ENV_PATH,
    PROCESSED_DIR,
    BATCH_DIR,
    CHUNKS_DIR,
    CHUNK_MODEL,
    MAX_BATCH_MSGS,
    CHUNK_BATCH_SIZE,
)
from utils import load_json_data, write_json_data

BATCH_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(dotenv_path=ENV_PATH)
client = openai.OpenAI()


def parse_timestamp(timestamp):
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
    return formatted


def format_batch_prompt(msgs):
    formatted = []
    for msg in msgs:
        content = msg.get("content", "").strip()
        if not content:
            continue
        time_str = parse_timestamp(msg["timestamp"])
        formatted.append(
            f"[{msg['message_id']}] ({time_str}) {msg['author_name']}: {content}"
        )

    return "\n".join(formatted)


def create_jsonl_requests(msgs):
    requests = []
    system_prompt = (
        "You are given a chronological chat log.\n"
        "Your task is to divide it into coherent conversation chunks, grouped by topic and extract a max of 5 keywords for each group (shorter conversations can have less keywords).\n"
        "Each message is prefixed with its ID in square brackes like [0123].\n"
        "IMPORTANT:\n"
        " - DO NOT repeat messages or write summaries.\n"
        " - DO NOT use the names of the people talking as keywords, as those are handled separately, names of other people mentioned in the conversation is ok.\n"
        " - DO NOT skip any messages, make sure every message ID is account for in exactly one chunk and in order.\n"
        "Return only a JSON list. Each list item is a JSON object with two keys:\n"
        "1. `keywords`: a list of relevant keywords\n"
        "2. `message_ids`: a list of message IDs in that chunk\n\n"
        "Example output:\n"
        "[\n"
        '  {"keywords": ["car repair", "exes", "jessica"], "message_ids": ["0123", "0124"]},\n'
        '  {"keywords": ["valorant"], "message_ids": ["2123"]}\n'
        "]\n\n"
    )

    for i in range(0, len(msgs), CHUNK_BATCH_SIZE):
        batch = msgs[i : i + CHUNK_BATCH_SIZE]
        prompt = format_batch_prompt(batch)
        request = {
            "custom_id": f"batch_{i // CHUNK_BATCH_SIZE + 1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": CHUNK_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            },
        }
        requests.append(request)

    return requests


def validate_jsonl(jsonl_path):
    errors = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            try:
                _ = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Line {i}] JSON decode error: {e}")
                errors += 1

    if errors == 0:
        print(f"{jsonl_path.name}: All lines are valid")
    else:
        print(f"{jsonl_path.name}: {errors} invalid lines found")


def submit_batch_request(jsonl_path, retries=3):
    print(f"Submitting batch job: {jsonl_path.name}")
    if not jsonl_path.exists():
        print(f"[ERROR] {jsonl_path.name} not found")
        return

    batch_input_file = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")

    try:
        response = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"source": jsonl_path.stem},
        )
        print(f"Submitted batch job with ID: {response.id}")
        wait_for_batch(response.id)
        save_batch_output(response.id)
    except Exception as e:
        print(f"Failed to submit batch: {e}")
        if retries > 0:
            print("Waiting 5 minutes to clear up queue then retrying...")
            time.sleep(5 * 60)
            submit_batch_request(jsonl_path, retries - 1)


def wait_for_batch(batch_id, poll_interval=30):
    start_time = time.time()
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status == "completed":
            break
        elif batch.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch {batch_id} failed with status {batch.status}")
        print(f"Batch {batch_id} still in progress... waiting {poll_interval}s")
        time.sleep(poll_interval)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Batch {batch_id} completed after {elapsed / 60:.2f} minutes")


def save_batch_output(batch_id):
    batch = client.batches.retrieve(batch_id)
    output_file_id = batch.output_file_id
    batch_file = batch.metadata["source"]
    input_file = re.sub(r"_batch_\d+$", "", batch_file) + ".json"
    input_path = PROCESSED_DIR / input_file
    msgs = load_json_data(input_path)
    msgs_by_id = {msg["message_id"]: msg for msg in msgs}

    if not output_file_id:
        print("Batch has no output file yet")
        return

    file_response = client.files.content(output_file_id).read().decode("utf-8")
    lines = file_response.strip().splitlines()

    chunks = []

    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
            body = record.get("response", {}).get("body", "")
            if isinstance(body, str):
                body = json.loads(body)
            content = body["choices"][0]["message"]["content"]
            conversations = json.loads(content)
        except Exception as e:
            print(f"Error in line {i}: {e}")
            continue

        for conv in conversations:
            msg_ids = conv["message_ids"]
            matched_msgs = [msgs_by_id[id] for id in msg_ids if id in msgs_by_id]
            if not matched_msgs:
                continue

            chunks.append(
                {
                    "chunk_id": f"{batch_id}_{i}",
                    "keywords": conv.get("keywords", []),
                    "messages": matched_msgs,
                    "author_names": list({m["author_name"] for m in matched_msgs}),
                }
            )

    output_file = batch_file.replace("processed_batch", "chunked")
    output_path = CHUNKS_DIR / f"{output_file}.json"
    write_json_data(output_path, chunks)
    print(f"Saved {len(chunks)} conversations from batch {batch_file} to {output_path}")


def main():
    # batch messages into jsonl
    for file_path in PROCESSED_DIR.glob("*_processed.json"):
        batch_prefix = f"{file_path.stem}_batch_"
        msgs = load_json_data(file_path)

        for i in range(0, len(msgs), MAX_BATCH_MSGS):
            batch = msgs[i : i + MAX_BATCH_MSGS]
            output_path = BATCH_DIR / f"{batch_prefix}{i // MAX_BATCH_MSGS:02}.jsonl"

            if output_path.exists():
                print(f"Output already exists for {output_path.name}, skipping")
                continue

            requests = create_jsonl_requests(batch)

            with output_path.open("w", encoding="utf-8") as f:
                for request in requests:
                    f.write(json.dumps(request) + "\n")
            validate_jsonl(output_path)
            print(
                f"Created {len(requests)} requests for {file_path.name} in {output_path}\n"
            )

    # submit jsonl batches and save output
    for jsonl_path in BATCH_DIR.glob("*_batch*.jsonl"):
        batch_prefix = jsonl_path.stem.replace("processed_batch", "chunked")
        output_path = CHUNKS_DIR / f"{batch_prefix}.json"

        if output_path.exists():
            print(
                f"Skipping {jsonl_path.name} - output already exists at {output_path.name}"
            )
            continue

        submit_batch_request(jsonl_path)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nScript finished in {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
