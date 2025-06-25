import json


def load_json_data(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_data(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
