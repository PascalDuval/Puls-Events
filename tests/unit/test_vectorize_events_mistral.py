from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import pytest

import vectorize_events_mistral as vec


class _FakeEmbeddingItem:
    def __init__(self, embedding: List[float]) -> None:
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, vectors: List[List[float]]) -> None:
        self.data = [_FakeEmbeddingItem(v) for v in vectors]


class FakeMistralClient:
    def embeddings(self, model: str, input: List[str]) -> Any:  # noqa: A002
        vectors = []
        for text in input:
            text_len = float(len(text))
            vectors.append([text_len, 1.0, 2.0])
        return _FakeEmbeddingResponse(vectors)


def test_prepare_embedding_document_from_normalized_record() -> None:
    record = {
        "id": "evenements-publics-openagenda:abc",
        "title": "Concert de printemps",
        "content": "Titre: Concert de printemps\nResume: Programme symphonique\nVille: Paris\nTags: musique, concert",
        "event_start": "2026-05-01T18:00:00+00:00",
        "source_record_url": "https://openagenda.com/example/events/abc",
        "tags": ["musique", "concert"],
    }

    doc = vec.prepare_embedding_document(record=record, fallback_index=1, min_text_chars=20)

    assert doc is not None
    assert doc["id"] == "evenements-publics-openagenda:abc"
    assert "Concert de printemps" in doc["text"]
    assert doc["metadata"]["source_record_url"].startswith("https://")


def test_prepare_embedding_document_from_raw_record() -> None:
    record = {
        "uid": "12345",
        "title_fr": "Exposition photo",
        "description_fr": "Une exposition de photos historiques en centre-ville.",
        "firstdate_begin": "2026-06-10T09:00:00+00:00",
        "lastdate_end": "2026-06-10T17:00:00+00:00",
        "location_city": "Lyon",
        "location_region": "Auvergne-Rhone-Alpes",
        "canonicalurl": "https://openagenda.com/example/events/12345",
        "keywords_fr": ["culture", "exposition"],
    }

    doc = vec.prepare_embedding_document(record=record, fallback_index=99, min_text_chars=20)

    assert doc is not None
    assert doc["id"] == "evenements-publics-openagenda:12345"
    assert "Exposition photo" in doc["text"]
    assert doc["metadata"]["city"] == "Lyon"


def test_prepare_embedding_document_rejects_missing_url() -> None:
    record = {
        "uid": "no-url",
        "title_fr": "Evenement sans URL",
        "description_fr": "Description assez longue pour depasser le seuil minimal de texte.",
    }

    doc = vec.prepare_embedding_document(record=record, fallback_index=3, min_text_chars=20)

    assert doc is None


def test_vectorize_jsonl_end_to_end_with_fake_client(tmp_path: Path) -> None:
    input_path = tmp_path / "events.jsonl"
    output_path = tmp_path / "vectors.jsonl"

    records = [
        {
            "id": "evenements-publics-openagenda:1",
            "title": "Festival A",
            "content": "Titre: Festival A Resume: programmation culturelle et musicale sur plusieurs jours en ville.",
            "source_record_url": "https://openagenda.com/e/events/1",
            "event_start": "2026-05-11T10:00:00+00:00",
            "tags": ["festival"],
        },
        {
            "uid": "2",
            "title_fr": "Atelier B",
            "description_fr": "Atelier participatif en bibliotheque municipale avec intervenants specialises.",
            "canonicalurl": "https://openagenda.com/e/events/2",
            "firstdate_begin": "2026-07-01T08:00:00+00:00",
            "lastdate_end": "2026-07-01T10:00:00+00:00",
            "location_city": "Nantes",
        },
        {
            "uid": "3",
            "title_fr": "Invalide C",
            "description_fr": "Texte invalide car URL absente meme si la description est longue.",
        },
    ]

    with input_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = vec.vectorize_jsonl(
        input_path=str(input_path),
        output_path=str(output_path),
        api_key="fake-api-key",
        batch_size=2,
        min_text_chars=20,
        client=FakeMistralClient(),
    )

    assert stats["read"] == 3
    assert stats["prepared"] == 2
    assert stats["skipped"] == 1
    assert stats["embedded"] == 2
    assert stats["written"] == 2

    with output_path.open("r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]

    assert len(lines) == 2
    for line in lines:
        assert line["id"]
        assert isinstance(line["embedding"], list)
        assert len(line["embedding"]) == 3
        assert line["metadata"]["source_record_url"].startswith("https://")


def test_vectorize_jsonl_raises_when_no_valid_documents(tmp_path: Path) -> None:
    input_path = tmp_path / "invalid.jsonl"
    output_path = tmp_path / "vectors.jsonl"

    input_path.write_text(
        json.dumps({"uid": "x", "title_fr": "Sans URL", "description_fr": "Description longue"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Aucun document exploitable"):
        vec.vectorize_jsonl(
            input_path=str(input_path),
            output_path=str(output_path),
            api_key="fake-api-key",
            min_text_chars=20,
            client=FakeMistralClient(),
        )


def test_resolve_api_key_from_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("MISTRAL_API_KEY=test-key-from-file\n", encoding="utf-8")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    resolved = vec.resolve_api_key(explicit_api_key=None, env_file=str(env_path))

    assert resolved == "test-key-from-file"
