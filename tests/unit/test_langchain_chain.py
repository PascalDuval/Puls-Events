"""Tests de la couche LangChain : embeddings et generation via les wrappers ChatMistralAI."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

from langchain_core.messages import HumanMessage, SystemMessage

from rag_chatbot_mistral import MistralRAGChatbot


class _FakeEmbeddings:
    def embed_query(self, text: str) -> List[float]:
        return [0.1] * 1024


class _FakeLLM:
    def __init__(self, answer: str = "Reponse.") -> None:
        self.answer = answer
        self.calls: List[Any] = []

    def invoke(self, messages: Any) -> Any:
        self.calls.append(messages)
        return SimpleNamespace(content=self.answer)


class _FakeSearcher:
    metadata: list = []

    def search_hybrid(self, *args: Any, **kwargs: Any) -> list:
        return []


def _make_bot(answer: str = "Reponse.") -> tuple:
    llm = _FakeLLM(answer=answer)
    embeddings = _FakeEmbeddings()
    bot = MistralRAGChatbot(embeddings=embeddings, llm=llm, searcher=_FakeSearcher())
    return bot, embeddings, llm


def test_embed_returns_list_of_floats() -> None:
    """_embed() doit retourner une liste de floats de longueur non nulle."""
    bot, _, _ = _make_bot()
    result = bot._embed("test requete")
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(v, float) for v in result)


def test_generate_returns_string() -> None:
    """_generate() doit retourner une chaine non vide."""
    bot, _, _ = _make_bot(answer="Voici une reponse concrete.")
    result = bot._generate("Ma question", "Mon contexte RAG")
    assert isinstance(result, str)
    assert result == "Voici une reponse concrete."


def test_generate_passes_system_and_human_messages() -> None:
    """_generate() doit passer un SystemMessage et un HumanMessage au LLM."""
    bot, _, llm = _make_bot()
    bot._generate("Quelle sortie ce week-end ?", "Doc 1: Festival jazz Paris.")

    assert len(llm.calls) == 1
    messages = llm.calls[0]
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "Tu es un assistant RAG" in messages[0].content
    assert "Quelle sortie ce week-end ?" in messages[1].content
    assert "Doc 1: Festival jazz Paris." in messages[1].content
