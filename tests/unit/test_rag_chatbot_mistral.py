from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from rag_chatbot_mistral import MistralRAGChatbot


@dataclass
class _FakeEmbeddingData:
    embedding: List[float]


class FakeMistralClient:
    def __init__(self, answer_text: str = "Reponse test.") -> None:
        self.answer_text = answer_text
        self.embedding_calls: List[Dict[str, Any]] = []
        self.chat_calls: List[Dict[str, Any]] = []

    def embeddings(self, model: str, input: List[str]) -> Any:
        self.embedding_calls.append({"model": model, "input": input})
        return SimpleNamespace(data=[_FakeEmbeddingData(embedding=[0.01] * 1024)])

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Any:
        self.chat_calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.answer_text))]
        )


class FakeSearcher:
    def __init__(self, docs: List[Dict[str, Any]]) -> None:
        self.docs = docs
        self.metadata = docs
        self.calls: List[Dict[str, Any]] = []

    def search_hybrid(
        self,
        query_embedding: List[float],
        k: int = 5,
        city: str | None = None,
        region: str | None = None,
        tags: List[str] | None = None,
        after_date: str | None = None,
        before_date: str | None = None,
    ) -> List[Dict[str, Any]]:
        self.calls.append(
            {
                "embedding_dim": len(query_embedding),
                "k": k,
                "city": city,
                "region": region,
                "tags": tags,
                "after_date": after_date,
                "before_date": before_date,
            }
        )
        return self.docs


def _sample_doc() -> Dict[str, Any]:
    return {
        "id": "evenements-publics-openagenda:123",
        "title": "Festival Jazz Paris",
        "city": "Paris",
        "region": "Ile-de-France",
        "event_start": "2026-05-20T18:00:00+00:00",
        "event_end": "2026-05-20T22:00:00+00:00",
        "tags": ["musique", "festival", "jazz"],
        "text_preview": "Concert jazz en plein air.",
        "source_record_url": "https://openagenda.com/events/123",
    }


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_rag_no_documents_returns_guardrail_message() -> None:
    """Sans documents recuperes, retourne un message sans appeler Mistral."""
    client = FakeMistralClient()
    searcher = FakeSearcher(docs=[])
    bot = MistralRAGChatbot(client=client, searcher=searcher)

    result = bot.ask("Quels concerts a Paris ?")

    assert "Je ne trouve pas" in result.answer
    assert result.sources == []
    assert len(result.documents) == 0
    assert len(client.chat_calls) == 0


def test_rag_calls_mistral_with_correct_parameters() -> None:
    """Verifie que mistral-small-latest est appele avec les bons parametres."""
    client = FakeMistralClient(answer_text="Voici une recommandation.\n\nSources:\n- https://openagenda.com/events/123")
    searcher = FakeSearcher(docs=[_sample_doc()])
    bot = MistralRAGChatbot(client=client, searcher=searcher)

    result = bot.ask("Propose un evenement musique a Paris.")

    assert result.model == "mistral-small-latest"
    assert len(client.embedding_calls) == 1
    assert len(client.chat_calls) == 1
    chat = client.chat_calls[0]
    assert chat["model"] == "mistral-small-latest"
    assert chat["temperature"] == pytest.approx(0.2)
    assert chat["top_p"] == pytest.approx(0.9)
    assert chat["max_tokens"] == 600


def test_rag_prompt_contains_system_rules_and_context() -> None:
    """Verifie que le prompt envoye a Mistral contient les regles et le contexte."""
    client = FakeMistralClient(answer_text="Reponse courte")
    searcher = FakeSearcher(docs=[_sample_doc()])
    bot = MistralRAGChatbot(client=client, searcher=searcher)

    bot.ask("Que faire a Paris ?", k=4, city="Paris", tags=["musique"])

    assert searcher.calls[0]["k"] == 4
    assert searcher.calls[0]["city"] == "Paris"
    assert searcher.calls[0]["tags"] == ["musique"]

    messages = client.chat_calls[0]["messages"]
    assert any("Tu es un assistant RAG" in m["content"] for m in messages)
    assert any("Cite explicitement les sources" in m["content"] for m in messages)
    assert any("https://openagenda.com/events/123" in m["content"] for m in messages)


def test_rag_appends_sources_when_llm_forgets_them() -> None:
    """Verifie que les sources sont ajoutees si le LLM ne les inclut pas."""
    client = FakeMistralClient(answer_text="Je recommande le festival de jazz.")
    searcher = FakeSearcher(docs=[_sample_doc()])
    bot = MistralRAGChatbot(client=client, searcher=searcher)

    result = bot.ask("Festival a Paris ?")

    assert "Sources:" in result.answer
    assert "https://openagenda.com/events/123" in result.answer
    assert result.sources == ["https://openagenda.com/events/123"]


def test_rag_does_not_duplicate_sources_already_in_answer() -> None:
    """Verifie que les sources ne sont pas dupliquees, meme en markdown gras."""
    answer_with_sources = (
        "Je recommande le festival.\n\n**Sources :**\n"
        "- https://openagenda.com/events/123"
    )
    client = FakeMistralClient(answer_text=answer_with_sources)
    searcher = FakeSearcher(docs=[_sample_doc()])
    bot = MistralRAGChatbot(client=client, searcher=searcher)

    result = bot.ask("Festival a Paris ?")

    assert result.answer.count("https://openagenda.com/events/123") == 1


def test_rag_infers_tags_from_question_when_none_provided() -> None:
    """Sans tags utilisateur, le bot infere des tags depuis la question."""
    client = FakeMistralClient(answer_text="Reponse test")
    searcher = FakeSearcher(docs=[_sample_doc()])
    bot = MistralRAGChatbot(client=client, searcher=searcher)

    bot.ask("As-tu un concert de jazz a Paris ?")

    assert len(searcher.calls) == 1
    inferred = searcher.calls[0]["tags"]
    assert inferred is not None
    assert "jazz" in inferred or "musique" in inferred


def test_rag_fallback_out_of_scope_skips_retrieval_and_llm() -> None:
    """Question hors contexte (meteo) -> fallback immediat sans retrieval."""
    client = FakeMistralClient(answer_text="Reponse test")
    searcher = FakeSearcher(docs=[_sample_doc()])
    bot = MistralRAGChatbot(client=client, searcher=searcher)

    result = bot.ask("Quel temps fait-il aujourd'hui ?")

    assert "hors du perimetre" in result.answer
    assert result.documents == []
    assert len(searcher.calls) == 0
    assert len(client.embedding_calls) == 0
    assert len(client.chat_calls) == 0


def test_rag_fallback_quantitative_skips_retrieval_and_llm() -> None:
    """Question quantitative sur la base -> fallback immediat sans retrieval."""
    client = FakeMistralClient(answer_text="Reponse test")
    searcher = FakeSearcher(docs=[_sample_doc()])
    bot = MistralRAGChatbot(client=client, searcher=searcher)

    result = bot.ask("Combien d'evenements sont dans la base de donnees ?")

    assert "statistiques globales" in result.answer
    assert result.documents == []
    assert len(searcher.calls) == 0
    assert len(client.embedding_calls) == 0
    assert len(client.chat_calls) == 0
