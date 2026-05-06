from __future__ import annotations

from dataclasses import dataclass
import inspect
import re
from typing import Any, Dict, List, Optional
import unicodedata

from mistralai.client import MistralClient

from faiss_searcher import FAISSSearcher
from utils.config import MISTRAL_API_KEY
from utils.temporal_deixis import infer_temporal_window


DEFAULT_LLM_MODEL = "mistral-small-latest"
DEFAULT_EMBEDDING_MODEL = "mistral-embed"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 600
DEFAULT_TOP_K = 20


def _supports_intent_tags(searcher: Any) -> bool:
    """Retourne True si search_hybrid accepte intent_tags."""
    try:
        signature = inspect.signature(searcher.search_hybrid)
    except (TypeError, ValueError, AttributeError):
        return False
    return "intent_tags" in signature.parameters

SYSTEM_PROMPT = (
    "Tu es un assistant RAG specialise dans les recommandations d'evenements culturels en Ile-de-France. "
    "REGLES STRICTES :"
    " 1) Utilise UNIQUEMENT les informations presentes dans les documents du contexte fourni."
    " 2) N'ajoute AUCUN detail, date, lieu ou titre qui ne figure pas explicitement dans le contexte."
    " 3) Si une information n'est pas disponible dans le contexte, ecris 'Information non disponible dans les documents.'"
    " 4) Ne formule AUCUNE hypothese sur ce qui pourrait exister hors du contexte."
    " 5) Cite explicitement les sources (URL) en fin de reponse pour chaque evenement mentionne."
)

FALLBACK_OUT_OF_SCOPE = (
    "Cette question semble hors du perimetre du bot (evenements culturels en Ile-de-France). "
    "Je peux t'aider pour des sorties culturelles (concert, exposition, spectacle, activites famille, etc.)."
)

FALLBACK_QUANTITATIVE = (
    "Je ne peux pas fournir de statistiques globales fiables sur la base (comptage total, volumes, KPI) "
    "via cette interface conversationnelle. Je peux en revanche recommander des evenements concrets."
)

FALLBACK_TOO_BROAD = (
    "Ta demande est trop generale pour que je puisse selectionner des evenements pertinents. "
    "Je donne de bien meilleurs resultats sur des requetes precises : un type d evenement, une ville ou une periode. "
    "Exemples : 'concert de jazz a Paris ce week-end', 'exposition photo en mai a Versailles', 'spectacle famille a Montreuil'."
)

# Patterns de requetes trop larges (sans filtre specifique type/lieu/date)
TOO_BROAD_PATTERNS = [
    r"\btous\s+les\s+(?:evenements?|spectacles?|concerts?|expositions?|festivals?|activites?)",
    r"\bl.ensemble\s+des\s+evenements?",
    r"\btout\s+le\s+programme",
    r"\btoute\s+la\s+programmation",
    r"\bliste\s+(?:complete|de\s+tous|exhaustive)",
    r"\btous\s+(?:les\s+)?evenements?\s+culturels?",
]

INTENT_TO_TAGS: Dict[str, List[str]] = {
    # Musique
    "jazz": ["jazz", "musique", "concert"],
    "concert": ["concert", "musique"],
    "musique": ["musique", "concert"],
    "festival": ["festival", "musique"],
    # Spectacle vivant
    "spectacle": ["spectacle", "theatre", "danse"],
    "theatre": ["theatre", "spectacle"],
    "danse": ["danse", "spectacle"],
    "cirque": ["cirque", "spectacle"],
    "humour": ["humour", "spectacle"],
    # Arts visuels
    "exposition": ["exposition", "art"],
    "photo": ["photo", "exposition"],
    "musee": ["musee", "exposition", "art"],
    "art": ["art", "exposition"],
    # Public famille/jeune
    "famille": ["famille", "enfant"],
    "enfant": ["enfant", "famille"],
    "ados": ["jeune public", "adolescent", "famille"],
    "ado": ["jeune public", "adolescent", "famille"],
    "adolescent": ["adolescent", "jeune public"],
    "jeune": ["jeune public", "adolescent"],
    "bebe": ["bebe", "enfant", "famille"],
    # Gratuit
    "gratuit": ["gratuit"],
    "libre": ["gratuit"],
    "entree libre": ["gratuit"],
    # Sortie générique
    "sortie": ["sortie", "famille", "loisir"],
    "balade": ["balade", "nature", "plein air"],
    "promenade": ["balade", "nature", "plein air"],
    # Thématique
    "science": ["science", "astronomie"],
    "astronomie": ["astronomie", "science"],
    "atelier": ["atelier"],
    "cinema": ["cinema"],
    "lecture": ["lecture", "litterature"],
    "litterature": ["litterature", "lecture"],
    "nature": ["nature", "plein air", "balade"],
    "plein air": ["plein air", "nature"],
    "patrimoine": ["patrimoine"],
    "gastronomie": ["gastronomie"],
    "sport": ["sport"],
}

OUT_OF_SCOPE_HINTS = [
    "meteo",
    "temps",
    "temperature",
    "pluie",
    "soleil",
    "neige",
    "humidite",
    "bourse",
    "crypto",
    "football",
    "politique",
]

CULTURE_GUARD_HINTS = [
    "evenement",
    "concert",
    "spectacle",
    "exposition",
    "festival",
    "sortie",
    "culture",
    "musee",
    "theatre",
    "danse",
    "activite",
]


@dataclass
class RAGAnswer:
    answer: str
    sources: List[str]
    documents: List[Dict[str, Any]]
    model: str
    inferred_tags: Optional[List[str]] = None
    effective_tags: Optional[List[str]] = None
    built_prompt: Optional[str] = None


class MistralRAGChatbot:
    """Chatbot RAG minimal : embed -> retrieve -> generate."""

    def __init__(
        self,
        index_dir: str = "data",
        api_key: Optional[str] = None,
        model: str = DEFAULT_LLM_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        searcher: Optional[FAISSSearcher] = None,
        client: Optional[MistralClient] = None,
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.searcher = searcher or FAISSSearcher(index_dir=index_dir, verbose=False)
        resolved_api_key = (api_key or MISTRAL_API_KEY).strip()
        if not resolved_api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY manquante. Definir la variable d'environnement "
                "ou renseigner un fichier .env local."
            )
        self.client = client or MistralClient(api_key=resolved_api_key)
        self.available_tags = self._build_available_tags()

    def ask(
        self,
        question: str,
        k: int = DEFAULT_TOP_K,
        city: Optional[str] = None,
        region: Optional[str] = None,
        tags: Optional[List[str]] = None,
        after_date: Optional[str] = None,
        before_date: Optional[str] = None,
    ) -> RAGAnswer:
        """Cycle RAG complet : embed -> retrieve -> generate."""
        question = (question or "").strip()
        if not question:
            raise ValueError("La question ne peut pas etre vide.")

        if self._is_out_of_scope_question(question):
            return RAGAnswer(
                answer=FALLBACK_OUT_OF_SCOPE,
                sources=[],
                documents=[],
                model=self.model,
            )

        if self._is_quantitative_database_question(question):
            return RAGAnswer(
                answer=FALLBACK_QUANTITATIVE,
                sources=[],
                documents=[],
                model=self.model,
            )

        if self._is_too_broad_question(question):
            return RAGAnswer(
                answer=FALLBACK_TOO_BROAD,
                sources=[],
                documents=[],
                model=self.model,
            )

        provided_tags = [self._normalize_tag(t) for t in (tags or []) if self._normalize_tag(t)]
        inferred_tags = self._infer_tags_from_question(question) if not provided_tags else []
        effective_tags = provided_tags or inferred_tags or None

        # Filtre temporel automatique depuis la deixis si non fourni explicitement
        if not after_date and not before_date:
            tw = infer_temporal_window(question)
            if tw:
                after_date = tw.after_date_utc
                before_date = tw.before_date_utc

        embedding = self._embed(question)

        search_kwargs = {
            "k": k,
            "city": city,
            "region": region,
            "tags": effective_tags,
            "after_date": after_date,
            "before_date": before_date,
        }
        if _supports_intent_tags(self.searcher):
            search_kwargs["intent_tags"] = effective_tags

        docs = self.searcher.search_hybrid(embedding, **search_kwargs)

        # Si l'inference de tags a ete trop restrictive, fallback sur retrieval sans tags.
        if not docs and inferred_tags and not provided_tags:
            fallback_search_kwargs = {
                "k": k,
                "city": city,
                "region": region,
                "tags": None,
                "after_date": after_date,
                "before_date": before_date,
            }
            if _supports_intent_tags(self.searcher):
                fallback_search_kwargs["intent_tags"] = None

            docs = self.searcher.search_hybrid(embedding, **fallback_search_kwargs)

        if not docs:
            return RAGAnswer(
                answer="Je ne trouve pas d'information exploitable dans les documents recuperes pour repondre a cette question.",
                sources=[],
                documents=[],
                model=self.model,
                inferred_tags=inferred_tags,
                effective_tags=list(effective_tags) if effective_tags else [],
            )

        context = self._format_context(docs)
        answer = self._generate(question, context)
        sources = self._extract_sources(docs)
        answer = self._append_sources_if_missing(answer, sources)
        built_prompt = f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n[USER]\nQuestion: {question}\n\nContexte:\n{context}"

        return RAGAnswer(
            answer=answer,
            sources=sources,
            documents=docs,
            model=self.model,
            inferred_tags=inferred_tags,
            effective_tags=list(effective_tags) if effective_tags else [],
            built_prompt=built_prompt,
        )

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> List[float]:
        response = self.client.embeddings(model=self.embedding_model, input=[text])
        return response.data[0].embedding

    def _generate(self, question: str, context: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nContexte:\n{context}"},
        ]
        response = self.client.chat(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _format_context(docs: List[Dict[str, Any]]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            title = doc.get("title", "Sans titre")
            city = doc.get("city", "")
            start = doc.get("event_start", "")
            preview = doc.get("text_preview", "")
            url = doc.get("source_record_url", "")
            parts.append(f"[{i}] {title} ({city}) - {start}\n{preview}\nSource: {url}")
        return "\n\n".join(parts)

    @staticmethod
    def _extract_sources(docs: List[Dict[str, Any]]) -> List[str]:
        seen: set = set()
        sources = []
        for doc in docs:
            url = doc.get("source_record_url", "")
            if url and url not in seen:
                seen.add(url)
                sources.append(url)
        return sources

    @staticmethod
    def _append_sources_if_missing(answer: str, sources: List[str]) -> str:
        if not sources:
            return answer
        # Detectionrobuste : "sources" peut apparaitre en gras, avec espace, etc.
        if any(url in answer for url in sources):
            return answer
        block = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)
        return answer + block

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text)
        normalized = "".join(c for c in normalized if not unicodedata.combining(c))
        return normalized.lower()

    def _build_available_tags(self) -> set[str]:
        available: set[str] = set()
        metadata = getattr(self.searcher, "metadata", [])
        for doc in metadata:
            for tag in doc.get("tags", []):
                norm = self._normalize_tag(tag)
                if norm:
                    available.add(norm)
        return available

    @staticmethod
    def _normalize_tag(tag: Any) -> str:
        text = str(tag or "").strip().lower()
        text = text.strip('"\'`“”‘’«»')
        text = re.sub(r"^#+", "", text)
        return text.strip()

    def _infer_tags_from_question(self, question: str) -> List[str]:
        if not self.available_tags:
            return []

        q_norm = self._normalize_text(question)
        tokens = [tok for tok in re.findall(r"[a-z0-9]+", q_norm) if len(tok) >= 3]
        candidates: List[str] = []

        for tok in tokens:
            if tok in self.available_tags:
                candidates.append(tok)

        for trigger, mapped_tags in INTENT_TO_TAGS.items():
            if trigger in q_norm:
                for tag in mapped_tags:
                    norm = self._normalize_tag(tag)
                    if norm in self.available_tags:
                        candidates.append(norm)

        # Dedup stable + limite pour eviter sur-filtrage.
        deduped = list(dict.fromkeys(candidates))
        return deduped[:6]

    def _is_out_of_scope_question(self, question: str) -> bool:
        q_norm = self._normalize_text(question)
        has_out_scope_hint = any(hint in q_norm for hint in OUT_OF_SCOPE_HINTS)
        has_culture_hint = any(hint in q_norm for hint in CULTURE_GUARD_HINTS)
        return has_out_scope_hint and not has_culture_hint

    def _is_quantitative_database_question(self, question: str) -> bool:
        q_norm = self._normalize_text(question)
        has_counting_intent = bool(re.search(r"\b(combien|nombre|total|statistiques?|kpi|volume)\b", q_norm))
        has_dataset_intent = bool(re.search(r"\b(evenements?|base|donnees?|dataset|corpus|index|faiss)\b", q_norm))
        return has_counting_intent and has_dataset_intent

    def _is_too_broad_question(self, question: str) -> bool:
        """Detecte les requetes trop generales sans filtre utile (type/lieu/date)."""
        q_norm = self._normalize_text(question)
        return any(re.search(pattern, q_norm) for pattern in TOO_BROAD_PATTERNS)
