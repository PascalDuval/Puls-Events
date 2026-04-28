"""Interface Streamlit conversationnelle pour Pull-Events IDF Bot.

Version alignée avec le bot RAG actuel:
- inférence des tags automatique (pas de filtre tags dans l'UI)
- dialogue multi-tours avec historique persistant en session
- indicateurs de fiabilité compréhensibles par requête
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from faiss_searcher import FAISSSearcher
from rag_chatbot_mistral import MistralRAGChatbot, RAGAnswer
from utils.temporal_deixis import infer_temporal_window


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="Pull-Events IDF Bot",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="⏳ Chargement de l'index FAISS...")
def load_faiss_searcher() -> FAISSSearcher:
    try:
        return FAISSSearcher(index_dir="data", verbose=False)
    except Exception as exc:
        st.error(f"❌ Erreur lors du chargement de l'index FAISS: {exc}")
        st.stop()


@st.cache_resource(show_spinner="⏳ Initialisation du chatbot RAG...")
def load_rag_chatbot() -> MistralRAGChatbot:
    try:
        return MistralRAGChatbot(index_dir="data")
    except Exception as exc:
        st.error(f"❌ Erreur lors de l'initialisation du chatbot: {exc}")
        st.stop()


faiss_searcher = load_faiss_searcher()
rag_chatbot = load_rag_chatbot()


def init_state() -> None:
    if "chat_messages" not in st.session_state:
        # Compatibilite avec les anciennes versions de l'app.
        st.session_state.chat_messages = []
    if "turns" not in st.session_state:
        st.session_state.turns = []
    if "prefill_prompt" not in st.session_state:
        st.session_state.prefill_prompt = ""
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""


def build_contextual_query(new_question: str) -> str:
    """Mode recherche simple: la requete envoyee est la requete saisie."""
    return new_question


def build_user_note(answer: RAGAnswer) -> str | None:
    """Ajoute une aide lisible si la reponse indique un match approximatif ou incomplet."""
    text = (answer.answer or "").lower()
    if not answer.documents:
        return "Aucun resultat suffisamment pertinent n'a ete trouve. Essaie de retirer une contrainte de date ou de reformuler plus simplement."
    if "information non disponible dans les documents" in text:
        return (
            "Le moteur a trouve des documents proches, mais pas une correspondance exacte sur tous les criteres. "
            "Tu peux essayer une version moins restrictive de la requete, par exemple sans date ou avec une ville seulement."
        )
    return None


def format_iso_date_fr(value: str | None, include_time: bool = False) -> str:
    """Convertit une date ISO vers un affichage français."""
    text = (value or "").strip()
    if not text:
        return "N/A"

    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if include_time:
            return dt.strftime("%d/%m/%Y %H:%M")
        return dt.strftime("%d/%m/%Y")
    except ValueError:
        pass

    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        yyyy, mm, dd = m.groups()
        return f"{dd}/{mm}/{yyyy}"

    return text


def has_explicit_time(value: str | None) -> bool:
    """Retourne True si la date ISO contient une heure explicite non nulle."""
    text = (value or "").strip()
    if not text:
        return False
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return any([dt.hour, dt.minute, dt.second])
    except ValueError:
        return False


def format_text_dates_fr(text: str) -> str:
    """Remplace les dates ISO YYYY-MM-DD dans un texte par DD/MM/YYYY."""
    if not text:
        return text

    def _replace(match: re.Match[str]) -> str:
        yyyy, mm, dd = match.group(1), match.group(2), match.group(3)
        return f"{dd}/{mm}/{yyyy}"

    return re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", _replace, text)


def format_event_period_fr(start_iso: str | None, end_iso: str | None) -> str:
    start = format_iso_date_fr(start_iso, include_time=has_explicit_time(start_iso))
    end = format_iso_date_fr(end_iso, include_time=has_explicit_time(end_iso))
    if start != "N/A" and end != "N/A":
        return f"du {start} au {end}"
    if start != "N/A":
        return f"a partir du {start}"
    if end != "N/A":
        return f"jusqu'au {end}"
    return "N/A"


def compute_reliability(answer: RAGAnswer, temporal_label: str | None) -> Dict[str, Any]:
    """Calcule des indicateurs de fiabilité lisibles pour l'utilisateur final."""
    docs = answer.documents or []
    doc_count = len(docs)
    unique_sources = len(set(answer.sources or []))

    scores: List[float] = []
    for doc in docs:
        score = doc.get("rerank_score", doc.get("similarity_score"))
        if isinstance(score, (int, float)):
            scores.append(float(score))
    avg_score = sum(scores) / len(scores) if scores else 0.0

    has_dates = 0
    for doc in docs:
        if doc.get("event_start") or doc.get("event_end"):
            has_dates += 1
    date_coverage = (has_dates / doc_count) if doc_count else 0.0

    answer_lower = (answer.answer or "").lower()
    is_fallback = (
        "hors du perimetre du bot" in answer_lower
        or "statistiques globales fiables" in answer_lower
        or "trop generale" in answer_lower
    )

    if is_fallback:
        level = "Hors perimetre"
        color = "⚪"
    elif doc_count == 0:
        level = "Faible"
        color = "🔴"
    elif avg_score >= 0.72 and unique_sources >= 2 and date_coverage >= 0.7:
        level = "Elevee"
        color = "🟢"
    elif avg_score >= 0.60 and unique_sources >= 1:
        level = "Moyenne"
        color = "🟠"
    else:
        level = "Faible"
        color = "🔴"

    return {
        "label": f"{color} {level}",
        "documents": doc_count,
        "sources": unique_sources,
        "avg_score": avg_score,
        "date_coverage": date_coverage,
        "temporal_label": temporal_label or "Aucune",
        "inferred_tags": answer.inferred_tags or [],
        "effective_tags": answer.effective_tags or [],
    }


def render_reliability_panel(rel: Dict[str, Any]) -> None:
    st.markdown("**Fiabilite de la reponse**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Niveau", rel["label"])
    with c2:
        st.metric("Docs utilises", rel["documents"])
    with c3:
        st.metric("Sources distinctes", rel["sources"])
    with c4:
        st.metric("Score retrieval moyen", f"{rel['avg_score']:.3f}")
    with c5:
        st.metric("Couverture dates", f"{rel['date_coverage'] * 100:.0f}%")

    # Fenetre temporelle
    tw = rel["temporal_label"]
    if tw and tw != "Aucune":
        st.caption(f"🕐 Fenetre temporelle detectee : **{tw}** (filtrage actif sur les dates des evenements)")
    else:
        st.caption("🕐 Fenetre temporelle : aucune expression de date detectee dans la question — pas de filtre temporel applique.")

    # Tags actifs dans la recherche hybride
    effective = rel.get("effective_tags", [])
    inferred = rel.get("inferred_tags", [])
    if effective:
        tags_display = "  ".join(f"`{t}`" for t in effective)
        st.caption(f"🏷️ Tags actifs dans la recherche hybride : {tags_display}")
        if inferred:
            st.caption(
                "Ces tags ont ete **inferres automatiquement** depuis ta question "
                "et utilises pour pre-filtrer les documents avant la recherche vectorielle."
            )
    else:
        st.caption(
            "🏷️ Aucun tag reconnu dans la question — la recherche s'est appuyee uniquement "
            "sur la similarite semantique (vecteur), sans filtre de tags."
        )


def render_sources(docs: List[Dict[str, Any]]) -> None:
    if not docs:
        st.info("Aucun document n'a ete retenu pour cette reponse.")
        return

    with st.expander("Voir les documents utilises", expanded=False):
        for i, doc in enumerate(docs, 1):
            title = doc.get("title", "Sans titre")
            city = doc.get("city", "N/A")
            region = doc.get("region", "N/A")
            period = format_event_period_fr(doc.get("event_start"), doc.get("event_end"))
            score = doc.get("rerank_score", doc.get("similarity_score", 0.0))
            url = doc.get("source_record_url", "")
            tags = ", ".join(doc.get("tags", [])[:5])

            st.markdown(f"**{i}. {title}**")
            st.caption(f"📍 {city}, {region} | 📅 {period} | score={float(score):.3f}")
            if tags:
                st.caption(f"🏷️ {tags}")
            if url:
                st.markdown(f"[Source]({url})")
            st.divider()


init_state()

with st.sidebar:
    st.title("⚙️ Reglages")
    st.subheader("Modele")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Controle la creativite du LLM lors de la generation de reponse. "
             "0.0-0.3 : reponse tres factuelle, collee aux documents. "
             "0.7-1.0 : reponse plus variee mais potentiellement moins fidele aux faits. "
             "Recommande : 0.1 a 0.3 pour ce bot.",
    )
    k_documents = st.slider(
        "Nombre de documents consultes",
        min_value=1,
        max_value=20,
        value=6,
        step=1,
        help="Nombre de documents retenus apres filtrage FAISS et passe au LLM comme contexte. "
             "Plus k est eleve = plus le LLM dispose d'informations = reponses potentiellement plus riches. "
             "Mais au-dela de 8-10, le LLM peut etre noye par trop de contexte. Recommande : 5 a 8.",
    )

    st.caption(
        "Conseil: pour des reponses tres factuelles, gardez une temperature basse "
        "(0.1 a 0.3) et 5 a 8 documents."
    )

    st.divider()
    st.subheader("Conversation")
    if st.button("Nouvelle conversation", use_container_width=True):
        st.session_state.turns = []
        st.session_state.prefill_prompt = ""
        st.session_state.query_input = ""
        st.rerun()

    if st.session_state.turns:
        st.caption(f"Tours: {len(st.session_state.turns)}")
        last = st.session_state.turns[-1]
        st.caption(f"Derniere requete: {last['timestamp']}")
    else:
        st.caption("Aucun echange pour le moment.")

    st.divider()
    stats = faiss_searcher.get_stats()
    st.subheader("Index")
    st.metric("Documents indexes", f"{stats.get('total_documents', 0):,}")
    st.metric("Villes uniques", stats.get("unique_cities", 0))
    st.metric("Tags uniques", stats.get("unique_tags", 0))


st.title("🎭 Pull-Events IDF Bot")
st.markdown(
    "Saisissez une demande simple et precise. Les exemples ci-dessous pre-remplissent des requetes qui fonctionnent sur le corpus actuel."
)

if not st.session_state.turns:
    st.info(
        "Exemples de demandes testes sur le corpus actuel: concert de jazz a Paris, "
        "ou exposition de photos en mai."
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🎷 Concert de jazz a Paris", use_container_width=True):
            st.session_state.query_input = "concert de jazz a Paris"
    with c2:
        if st.button("📸 Exposition de photos en mai", use_container_width=True):
            st.session_state.query_input = "exposition de photos en mai"

with st.form("search_form", clear_on_submit=False):
    user_prompt = st.text_input(
        "Votre recherche",
        key="query_input",
        placeholder="Ex: concert de jazz a Paris",
        label_visibility="collapsed",
    )
    search_clicked = st.form_submit_button("Lancer la recherche", use_container_width=True, type="primary")

if st.session_state.turns:
    st.divider()
    st.subheader("Dernier resultat")
    last_turn = st.session_state.turns[-1]
    st.markdown(f"**Requete:** {last_turn['question']}")
    st.markdown(format_text_dates_fr(last_turn["answer"]))
    if last_turn.get("user_note"):
        st.info(last_turn["user_note"])
    render_reliability_panel(last_turn["reliability"])
    render_sources(last_turn["documents"])

    if last_turn.get("built_prompt"):
        with st.expander("🔬 Prompt envoyé au LLM", expanded=False):
            st.code(last_turn["built_prompt"], language="text")

    with st.expander("Historique des recherches", expanded=False):
        for index, turn in enumerate(reversed(st.session_state.turns[:-1]), 1):
            st.markdown(f"**{index}. {turn['question']}**")
            if turn.get("user_note"):
                st.caption(turn["user_note"])
            st.caption(turn["timestamp"])
            st.divider()

if search_clicked and user_prompt.strip():
    with st.spinner("Recherche et generation en cours..."):
        try:
            logger.info("Nouvelle requete utilisateur: %s", user_prompt)
            chatbot_temp = MistralRAGChatbot(
                searcher=faiss_searcher,
                client=rag_chatbot.client,
                temperature=temperature,
            )

            contextual_query = build_contextual_query(user_prompt.strip())
            temporal = infer_temporal_window(user_prompt.strip())

            answer = chatbot_temp.ask(
                question=contextual_query,
                k=k_documents,
            )

            reliability = compute_reliability(
                answer=answer,
                temporal_label=temporal.label if temporal else None,
            )
            user_note = build_user_note(answer)

            st.session_state.turns.append(
                {
                    "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "question": user_prompt.strip(),
                    "answer": answer.answer,
                    "contextual_question": contextual_query,
                    "documents": answer.documents,
                    "sources": answer.sources,
                    "reliability": reliability,
                    "user_note": user_note,
                    "built_prompt": answer.built_prompt or "",
                }
            )
            st.rerun()
        except Exception as exc:
            logger.exception("Erreur lors du traitement RAG")
            err = f"❌ Erreur lors du traitement: {exc}"
            st.session_state.turns.append(
                {
                    "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "question": user_prompt.strip(),
                    "answer": err,
                    "contextual_question": user_prompt.strip(),
                    "documents": [],
                    "sources": [],
                    "reliability": {
                        "label": "⚪ Hors perimetre",
                        "documents": 0,
                        "sources": 0,
                        "avg_score": 0.0,
                        "date_coverage": 0.0,
                        "temporal_label": "Aucune",
                    },
                    "user_note": "Une erreur technique a interrompu la recherche.",
                }
            )
            st.rerun()


st.divider()
with st.expander("📚 Comment le bot fonctionne", expanded=False):
    st.markdown(
        """
1. Votre demande est vectorisee (embedding Mistral 1024 dimensions), puis comparee au corpus de 8 014 evenements via FAISS.
2. Le moteur commence par extraire des tags metier depuis votre question (`jazz` → `jazz`, `musique`, `concert` ; `famille` → `famille`, `enfant` ; etc.). Ces tags servent a pre-filtrer les candidats.
3. La recherche hybride s'effectue en deux passes :
   - **Passe 1 (FAISS)** : recuperation des top-K×5 candidats les plus proches semantiquement (distance L2).
   - **Passe 2 (filtres metadonnees)** : elimination des candidats qui ne matchent pas la ville, la region, les tags ou la fenetre temporelle detectee.
   - **Reranking** : les survivants sont retries par score combine (`similarite + bonus tags + bonus ville`).
4. Les K meilleurs documents apres reranking sont passes au LLM comme contexte.
5. Le LLM genere la reponse en s'appuyant **uniquement** sur ces documents (5 regles anti-hallucination dans le prompt systeme).

La detection des tags est automatique. Vous n'avez pas besoin de choisir des tags manuellement.

**Requetes non supportees :**
- Questions hors domaine (meteo, bourse, sport non culturel) → message de hors-perimetre.
- Demandes de statistiques globales (`combien d'evenements...`) → message de refus.
- Requetes trop generales sans filtre (`tous les evenements culturels...`) → invitation a reformuler.
        """
    )

with st.expander("🔧 Construction du prompt envoye au LLM", expanded=False):
    st.markdown(
        """
### Ce qui est envoye au LLM a chaque requete

Le prompt transmis au modele Mistral se compose de deux parties :

**1. Prompt systeme (fixe)** — definit le comportement du LLM :
```
Tu es un assistant RAG specialise dans les recommandations d evenements culturels en Ile-de-France.
REGLES STRICTES :
  1) Utilise UNIQUEMENT les informations presentes dans les documents du contexte fourni.
  2) N ajoute AUCUN detail, date, lieu ou titre qui ne figure pas explicitement dans le contexte.
  3) Si une information n est pas disponible dans le contexte, ecris "Information non disponible dans les documents."
  4) Ne formule AUCUNE hypothese sur ce qui pourrait exister hors du contexte.
  5) Cite systematiquement les sources (URL) en fin de reponse pour chaque evenement mentionne.
```

**2. Message utilisateur (dynamique)** — construit a chaque requete :
```
Question: <votre question>

Contexte:
[1] Titre de l evenement (Ville) - 2026-05-10
Description / resume de l evenement...
Source: https://openagenda.com/...

[2] ...
[3] ...
```

### Role du slider « Nombre de documents consultes » (k)

Après le filtrage et le reranking, les **k meilleurs documents** sont injectes dans le contexte ci-dessus.
- k = 3 : contexte minimal, reponse courte et precise, risque de manquer des evenements pertinents.
- k = 6-8 : equilibre recommande entre richesse et coherence.
- k = 15-20 : contexte tres large, mais le LLM peut avoir du mal a synthetiser autant d information.

### Role du slider « Temperature »

La temperature controle la **creativite du LLM** au moment de la generation de la reponse :
- **0.0-0.3** : le LLM colle au texte des documents, reponse factuelle et previsible. Ideal pour les recommandations.
- **0.5-0.7** : plus de variete dans la formulation, mais la fidelite aux faits diminue.
- **0.8-1.0** : reponse tres creative, risque de paraphrase ou de deviation par rapport aux documents.

### Comment les tags participent a la recherche hybride

La recherche hybride utilise les tags en deux endroits :
1. **Filtre dur** : un document est elimine si aucun de ses tags ne figure dans les tags de la requete.
2. **Bonus de reranking** : `+0.15` si ≥ 2 tags matchent, `+0.08` si 1 seul tag matche. Cela remonte les documents tres pertinents au-dessus des documents semantiquement proches mais thematiquement moins cibles.

Si les tags sont trop restrictifs et produisent 0 resultat, un fallback automatique relance la recherche sans filtre de tags.
        """
    )

with st.expander("📏 Comment lire les indicateurs de fiabilite", expanded=False):
    st.markdown(
        """
- **Niveau** : estimation globale de robustesse de la reponse (Faible / Moyenne / Elevee / Hors perimetre).
  - 🟢 Elevee : score moyen ≥ 0.72, ≥ 2 sources distinctes, ≥ 70% des docs avec des dates.
  - 🟠 Moyenne : score moyen ≥ 0.60, au moins 1 source.
  - 🔴 Faible : trop peu de documents ou score trop bas.
  - ⚪ Hors perimetre : question hors domaine, trop generale ou statistique.
- **Docs utilises** : nombre de documents effectivement passes au LLM comme contexte (= valeur du slider k).
- **Sources distinctes** : nombre d URL OpenAgenda differentes parmi les documents. Plus ce nombre est eleve, plus la reponse est recoupee entre plusieurs evenements.
- **Score retrieval moyen** : proximite semantique moyenne (cosinus / L2 FAISS) entre votre requete et les documents retenus. Un score proche de 1.0 indique un tres bon alignement.
- **Couverture dates** : part des documents retenus qui contiennent des dates d evenement exploitables. Un taux eleve valide que le filtrage temporel a fonctionne.
- **Fenetre temporelle** : expression de date detectee dans votre question ("ce week-end", "en mai", etc.). Si detectee, elle a ete utilisee pour filtrer les evenements par leur date.
- **Tags actifs** : mots-cles metier inferences automatiquement depuis votre question et utilises comme filtre dur dans la recherche hybride.

Ces indicateurs aident a jauger la confiance, mais ne remplacent pas une verification humaine des liens source.
        """
    )
