import json
from config import RAW_DIR, CLEAN_DIR

CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def load_json_data(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_data(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def filter_data(data, author_id=None):
    msgs = data["messages"]
    filtered = []

    for msg in msgs:
        if msg["type"] != "Default" or msg["author"]["isBot"] is True:
            continue
        if author_id and msg["author"]["id"] != author_id:
            continue
        if not msg["content"].strip():
            continue

        filtered.append(
            {
                "message_id": msg["id"],
                "author_id": msg["author"]["id"],
                "content": msg["content"],
                "timestamp": msg["timestamp"],
            }
        )

    return filtered


def main():
    filtered_messages = []
    files = [f"general{i}.json" for i in range(10, 17)] + ["thelads.json"]
    print(f"Filtering {len(files)} files in {RAW_DIR}")

    for file_name in files:
        if file_name.endswith(".json"):
            file_path = RAW_DIR / file_name
            print(f"Processing file: {file_name}")
            try:
                filtered = filter_data(load_json_data(file_path))
                filtered_messages.extend(filtered)
                print(f"Filtered {len(filtered)} messages")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    filtered_messages.sort(key=lambda msg: msg["timestamp"])

    output_path = CLEAN_DIR / "filtered_messages.json"
    write_json_data(output_path, filtered_messages)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(
        f"Saved {len(filtered_messages)} messages to {output_path} ({size_mb:.2f} MB)"
    )


if __name__ == "__main__":
    main()
