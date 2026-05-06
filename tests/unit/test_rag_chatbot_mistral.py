from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from rag_chatbot_mistral import MistralRAGChatbot


class FakeMistralAIEmbeddings:
    def __init__(self) -> None:
        self.embed_query_calls: List[str] = []

    def embed_query(self, text: str) -> List[float]:
        self.embed_query_calls.append(text)
        return [0.01] * 1024


class FakeChatMistralAI:
    def __init__(self, answer_text: str = "Reponse test.") -> None:
        self.answer_text = answer_text
        self.invoke_calls: List[Any] = []

    def invoke(self, messages: Any) -> Any:
        self.invoke_calls.append(messages)
        return SimpleNamespace(content=self.answer_text)


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


def _make_bot(answer_text: str = "Reponse test.", docs: List[Dict[str, Any]] | None = None):
    embeddings = FakeMistralAIEmbeddings()
    llm = FakeChatMistralAI(answer_text=answer_text)
    searcher = FakeSearcher(docs=docs if docs is not None else [])
    bot = MistralRAGChatbot(embeddings=embeddings, llm=llm, searcher=searcher)
    return bot, embeddings, llm, searcher


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_rag_no_documents_returns_guardrail_message() -> None:
    """Sans documents recuperes, retourne un message sans appeler le LLM."""
    bot, embeddings, llm, searcher = _make_bot(docs=[])

    result = bot.ask("Quels concerts a Paris ?")

    assert "Je ne trouve pas" in result.answer
    assert result.sources == []
    assert len(result.documents) == 0
    assert len(llm.invoke_calls) == 0


def test_rag_calls_llm_with_correct_model() -> None:
    """Verifie que le modele configure est bien utilise."""
    bot, embeddings, llm, searcher = _make_bot(
        answer_text="Voici une recommandation.\n\nSources:\n- https://openagenda.com/events/123",
        docs=[_sample_doc()],
    )

    result = bot.ask("Propose un evenement musique a Paris.")

    assert result.model == "mistral-small-latest"
    assert len(embeddings.embed_query_calls) == 1
    assert len(llm.invoke_calls) == 1


def test_rag_prompt_contains_system_rules_and_context() -> None:
    """Verifie que le prompt envoye au LLM contient les regles et le contexte."""
    bot, embeddings, llm, searcher = _make_bot(answer_text="Reponse courte", docs=[_sample_doc()])

    bot.ask("Que faire a Paris ?", k=4, city="Paris", tags=["musique"])

    assert searcher.calls[0]["k"] == 4
    assert searcher.calls[0]["city"] == "Paris"
    assert searcher.calls[0]["tags"] == ["musique"]

    messages = llm.invoke_calls[0]
    assert any("Tu es un assistant RAG" in m.content for m in messages)
    assert any("Cite explicitement les sources" in m.content for m in messages)
    assert any("https://openagenda.com/events/123" in m.content for m in messages)


def test_rag_appends_sources_when_llm_forgets_them() -> None:
    """Verifie que les sources sont ajoutees si le LLM ne les inclut pas."""
    bot, _, _, _ = _make_bot(
        answer_text="Je recommande le festival de jazz.",
        docs=[_sample_doc()],
    )

    result = bot.ask("Festival a Paris ?")

    assert "Sources:" in result.answer
    assert "https://openagenda.com/events/123" in result.answer
    assert result.sources == ["https://openagenda.com/events/123"]


def test_rag_does_not_duplicate_sources_already_in_answer() -> None:
    """Verifie que les sources ne sont pas dupliquees."""
    answer_with_sources = (
        "Je recommande le festival.\n\n**Sources :**\n"
        "- https://openagenda.com/events/123"
    )
    bot, _, _, _ = _make_bot(answer_text=answer_with_sources, docs=[_sample_doc()])

    result = bot.ask("Festival a Paris ?")

    assert result.answer.count("https://openagenda.com/events/123") == 1


def test_rag_infers_tags_from_question_when_none_provided() -> None:
    """Sans tags utilisateur, le bot infere des tags depuis la question."""
    bot, _, _, searcher = _make_bot(answer_text="Reponse test", docs=[_sample_doc()])

    bot.ask("As-tu un concert de jazz a Paris ?")

    assert len(searcher.calls) == 1
    inferred = searcher.calls[0]["tags"]
    assert inferred is not None
    assert "jazz" in inferred or "musique" in inferred


def test_rag_fallback_out_of_scope_skips_retrieval_and_llm() -> None:
    """Question hors contexte (meteo) -> fallback immediat sans retrieval."""
    bot, embeddings, llm, searcher = _make_bot(docs=[_sample_doc()])

    result = bot.ask("Quel temps fait-il aujourd'hui ?")

    assert "hors du perimetre" in result.answer
    assert result.documents == []
    assert len(searcher.calls) == 0
    assert len(embeddings.embed_query_calls) == 0
    assert len(llm.invoke_calls) == 0


def test_rag_fallback_quantitative_skips_retrieval_and_llm() -> None:
    """Question quantitative sur la base -> fallback immediat sans retrieval."""
    bot, embeddings, llm, searcher = _make_bot(docs=[_sample_doc()])

    result = bot.ask("Combien d'evenements sont dans la base de donnees ?")

    assert "statistiques globales" in result.answer
    assert result.documents == []
    assert len(searcher.calls) == 0
    assert len(embeddings.embed_query_calls) == 0
    assert len(llm.invoke_calls) == 0

