from __future__ import annotations

import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_JSONL = PROJECT_ROOT / "data" / "evenements_publics_openagenda_culture_ile_de_france_rag.jsonl"


def main() -> None:
    random.seed(42)
    with INPUT_JSONL.open("r", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]

    samples = random.sample(events, 3)
    for i, ev in enumerate(samples, 1):
        print(f"--- Exemple {i} ---")
        print(f"title: {ev.get('title')}")
        print(f"location_name: {ev.get('location_name')}")
        print(f"city: {ev.get('city')}")
        print(f"region: {ev.get('region')}")
        print(f"event_start: {ev.get('event_start')}")
        print(f"tags: {ev.get('tags')}")
        summary = ev.get("summary", "")
        print(f"summary: {summary[:217] + '...' if len(summary) > 220 else summary}")
        print(f"source_record_url: {ev.get('source_record_url')}")
        print("")


if __name__ == "__main__":
    main()
