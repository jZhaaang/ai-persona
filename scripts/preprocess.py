import json
import os
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
SOURCE_DIR = (SCRIPTS_DIR / ".." / "data" / "raw").resolve()
OUTPUT_DIR = (SCRIPTS_DIR / ".." / "data" / "clean").resolve()

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_json_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


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


if __name__ == "__main__":
    filtered_messages = []
    files = [f"general{i}.json" for i in range(10, 17)] + ["thelads.json"]
    print(f"Filtering {len(files)} files in {SOURCE_DIR}")

    for file_name in files:
        if file_name.endswith(".json"):
            file_path = os.path.join(SOURCE_DIR, file_name)
            print(f"Processing file: {file_name}")
            try:
                filtered = filter_data(load_json_data(file_path))
                filtered_messages.extend(filtered)
                print(f"Filtered {len(filtered)} messages")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    filtered_messages.sort(key=lambda msg: msg["timestamp"])

    output_path = os.path.join(OUTPUT_DIR, "filtered_messages.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_messages, f, indent=2, ensure_ascii=False)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(
        f"Saved {len(filtered_messages)} messages to {output_path} ({size_mb:.2f} MB)"
    )
