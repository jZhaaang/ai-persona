import json
from config import RAW_DIR, PROCESSED_DIR, AUTHORS_PATH
from utils import load_json_data, write_json_data

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def trim_data(data):
    msgs = data["messages"]
    trimmed = []

    for msg in msgs:
        if msg["type"] != "Default" or msg["author"]["isBot"] is True:
            continue
        if not msg["content"].strip():
            continue

        trimmed.append(
            {
                "message_id": msg["id"],
                "author_id": msg["author"]["id"],
                "content": msg["content"],
                "timestamp": msg["timestamp"],
            }
        )

    return trimmed


def annotate_authors(msgs):
    authors_map = load_json_data(AUTHORS_PATH)
    for msg in msgs:
        msg["author_name"] = authors_map.get(msg["author_id"], "Unknown")

    return msgs


def process_file(input_path):
    raw_data = load_json_data(input_path)
    trimmed = trim_data(raw_data)
    processed = annotate_authors(trimmed)

    output_name = input_path.stem + "_processed.json"
    output_path = PROCESSED_DIR / output_name
    write_json_data(output_path, processed)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(processed)} messages to {output_path} ({size_mb:.2f} MB)")


def main():
    allowed_files = {f"general{n}" for n in range(10, 17)}
    allowed_files.add("thelads")

    input_files = [f for f in RAW_DIR.glob("*.json") if f.stem in allowed_files]
    print(f"Processing {len(input_files)} files in {RAW_DIR}")

    for input_path in input_files:
        print(f"Processing {input_path.name}...")
        process_file(input_path)


if __name__ == "__main__":
    main()
