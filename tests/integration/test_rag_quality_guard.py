"""Regression guard for RAG-ready JSONL quality."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import openagenda_culture_france_rag as rag


def _read_jsonl(path: Path) -> list[dict]:
    docs: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def test_generated_jsonl_is_vectorizable_and_consistent(tmp_path: Path) -> None:
    output_path = tmp_path / "quality_guard_sample.jsonl"

    # Reprise: fenêtre fixe 2025-04-25 -> 2027-04-25, IDF uniquement.
    rag.build_rag_file(output_path=str(output_path), max_records=400)

    assert output_path.exists(), "Le fichier JSONL temporaire n'a pas ete genere."
    docs = _read_jsonl(output_path)
    assert docs, "Aucun document genere: impossible de valider la qualite RAG."

    missing_required: list[str] = []
    invalid_dates: list[str] = []
    non_vectorizable: list[str] = []
    invalid_source_url: list[str] = []

    for doc in docs:
        doc_id = str(doc.get("id", "<missing-id>"))

        for key in ("id", "title", "content", "event_start", "event_end", "source_record_url", "metadata"):
            if not doc.get(key):
                missing_required.append(f"{doc_id}:{key}")

        try:
            start = datetime.fromisoformat(str(doc.get("event_start")))
            end = datetime.fromisoformat(str(doc.get("event_end")))
            if end < start:
                invalid_dates.append(doc_id)
        except Exception:
            invalid_dates.append(doc_id)

        if not rag.is_vectorizable(doc):
            non_vectorizable.append(doc_id)

        source_url = str(doc.get("source_record_url", ""))
        if not source_url.startswith(("http://", "https://")):
            invalid_source_url.append(doc_id)

        metadata = doc.get("metadata") or {}
        quality_missing_fields = metadata.get("quality_missing_fields")
        if quality_missing_fields is not None:
            assert isinstance(quality_missing_fields, list), (
                f"quality_missing_fields doit etre une liste (doc={doc_id})."
            )

    assert not missing_required, (
        "Champs requis manquants detectes: " + ", ".join(missing_required[:8])
    )
    assert not invalid_dates, (
        "Dates invalides ou incoherentes detectees: " + ", ".join(invalid_dates[:8])
    )
    assert not non_vectorizable, (
        "Documents non exploitables pour embeddings detectes: " + ", ".join(non_vectorizable[:8])
    )
    assert not invalid_source_url, (
        "URLs source invalides detectees: " + ", ".join(invalid_source_url[:8])
    )
