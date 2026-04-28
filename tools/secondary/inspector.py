from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_JSONL = PROJECT_ROOT / "data" / "test_vectors_50.jsonl"


def main() -> None:
    with INPUT_JSONL.open("r", encoding="utf-8") as handle:
        for _ in range(2):
            line = handle.readline()
            if not line:
                break
            record = json.loads(line)
            print(f"id: {record.get('id')}")
            print(f"text[:200]: {record.get('text', '')[:200]}")
            print(f"embedding[:3]: {record.get('embedding', [])[:3]}")
            print(f"metadata keys: {list(record.get('metadata', {}).keys())}")
            print()


if __name__ == "__main__":
    main()
