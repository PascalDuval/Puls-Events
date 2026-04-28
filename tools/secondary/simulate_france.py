from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openagenda_culture_france_rag import (
    OPENAGENDA_MAX_OFFSET,
    WINDOW_END_UTC,
    WINDOW_START_UTC,
    extract_dates,
    fetch_records,
    is_vectorizable,
    looks_cultural,
    normalize_record,
    overlaps_window,
)


def simulate_france_counts() -> None:
    where_clause = (
        "location_countrycode='FR'"
        f" AND firstdate_begin <= date'{WINDOW_END_UTC.date().isoformat()}'"
        f" AND lastdate_end >= date'{WINDOW_START_UTC.date().isoformat()}'"
    )

    fetched = 0
    kept_france = 0
    skipped_non_cultural = 0
    skipped_outside_window = 0
    skipped_not_vectorizable = 0
    skipped_duplicates = 0
    seen = set()

    for record in fetch_records(limit=100, where_clause=where_clause, max_offset=OPENAGENDA_MAX_OFFSET):
        fetched += 1

        if not looks_cultural(record):
            skipped_non_cultural += 1
            continue

        start_dt, end_dt = extract_dates(record)
        if not overlaps_window(start_dt, end_dt, WINDOW_START_UTC, WINDOW_END_UTC):
            skipped_outside_window += 1
            continue

        doc = normalize_record(record)
        if not is_vectorizable(doc):
            skipped_not_vectorizable += 1
            continue

        dedup_key = doc["metadata"]["raw_uid"] or (doc["title"].strip().lower(), doc["event_start"], doc["city"])
        if dedup_key in seen:
            skipped_duplicates += 1
            continue
        seen.add(dedup_key)

        kept_france += 1

    print(f"fetched: {fetched}")
    print(f"kept_france: {kept_france}")
    print(f"skipped_non_cultural: {skipped_non_cultural}")
    print(f"skipped_outside_window: {skipped_outside_window}")
    print(f"skipped_not_vectorizable: {skipped_not_vectorizable}")
    print(f"skipped_duplicates: {skipped_duplicates}")
    print(f"hit_offset_limit: {str(fetched >= OPENAGENDA_MAX_OFFSET).lower()}")


if __name__ == "__main__":
    simulate_france_counts()
