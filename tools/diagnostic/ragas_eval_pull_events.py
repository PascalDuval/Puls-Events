"""
Evaluation RAGAS pour Pull-Events (sans modifier les scripts existants).

Ce script:
1) Lance le RAG sur un jeu de questions representatif.
2) Capture les reponses et les contextes effectivement recuperes.
3) Genere une ground truth "silver" (alternative au labeling manuel).
4) Calcule les metriques RAGAS.
5) Affiche un diagnostic exploitable en console.

Usage:
  C:/Users/karap/anaconda3/envs/LLMRag/python.exe tools/diagnostic/ragas_eval_pull_events.py \
      --env-file "C:/Users/karap/OpenClassRooms/projet11/coursEtExos/8532116-mettez-en-place-un-rag-pour-un-llm/.env" \
      --k 6
"""

from __future__ import annotations

import argparse
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List

import nest_asyncio
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, context_utilization, faithfulness
# from ragas.metrics import answer_relevancy  # TODO: activer avec RAGAS >= 0.2.x (retourne NaN en 0.1.22 — TypeError: unsupported operand type(s) for +=: dict and dict)

# Import des modules du projet sans modifier PYTHONPATH global de l'environnement
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_chatbot_mistral import MistralRAGChatbot  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation RAGAS pour Pull-Events")
    parser.add_argument("--env-file", default="", help="Chemin vers un .env contenant MISTRAL_API_KEY")
    parser.add_argument("--k", type=int, default=6, help="Nombre de documents recuperes par question")
    parser.add_argument("--runs", type=int, default=1, help="Nombre de runs RAGAS (pour stabilite stat)")
    return parser.parse_args()


def build_test_questions() -> List[str]:
    return [
        "je cherche une sortie en famille ce week-end en ile de france",
        "as-tu un concert de jazz a Paris ?",
        "je veux une exposition photo en ile de france",
        "donne moi un evenement gratuit pour enfant",
        "je cherche une activite autour des sciences ou de l astronomie",
        "propose moi un spectacle a Aubervillier",
        "propose moi une activite gratuite pour ados en ile de france",
        # "Combien d'evenements gratuits en Idf et a quel public cela s'adresse-t-il ?"  # desormais gere par fallback quantitatif
        # "Bonjour, quel temps fait-il aujourd'hui ?"  # retiree : question volontairement hors-sujet, fausse les scores moyens
    ]


def doc_to_context(doc: Dict[str, Any]) -> str:
    title = doc.get("title", "Sans titre")
    city = doc.get("city", "")
    region = doc.get("region", "")
    start = doc.get("event_start", "")
    end = doc.get("event_end", "")
    preview = doc.get("text_preview", "")
    tags = ", ".join(doc.get("tags", [])[:8])
    url = doc.get("source_record_url", "")

    return (
        f"Titre: {title}\n"
        f"Ville: {city}\n"
        f"Region: {region}\n"
        f"Debut: {start}\n"
        f"Fin: {end}\n"
        f"Tags: {tags}\n"
        f"Apercu: {preview}\n"
        f"Source: {url}"
    )


def build_silver_ground_truth(llm: ChatMistralAI, question: str, contexts: List[str]) -> str:
    if not contexts:
        return "Aucune information exploitable n'a ete recuperee dans le corpus pour cette question."

    compact_context = "\n\n".join(contexts[:6])
    prompt = (
        "Tu produis une reference de correction (ground truth silver) pour evaluer un systeme RAG.\n"
        "Regles strictes:\n"
        "- Utilise uniquement le contexte ci-dessous.\n"
        "- N'invente rien.\n"
        "- Reponse concise (3-6 lignes max).\n"
        "- Si la question est hors sujet par rapport au contexte culturel, dis-le explicitement.\n\n"
        f"Question: {question}\n\n"
        f"Contexte:\n{compact_context}\n"
    )
    return llm.invoke(prompt).content.strip()


def run_rag_collection(chatbot: MistralRAGChatbot, questions: List[str], k: int) -> tuple[List[str], List[List[str]], List[List[Dict[str, Any]]]]:
    answers: List[str] = []
    contexts: List[List[str]] = []
    retrieved_docs: List[List[Dict[str, Any]]] = []

    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] RAG -> {question}")
        rag_answer = chatbot.ask(question=question, k=k, tags=None)
        answers.append(rag_answer.answer)
        retrieved_docs.append(rag_answer.documents)
        contexts.append([doc_to_context(d) for d in rag_answer.documents])

    return answers, contexts, retrieved_docs


def print_retrieval_overview(questions: List[str], docs_per_question: List[List[Dict[str, Any]]]) -> None:
    print("\n=== Apercu retrieval (inference intentions/tags active) ===")
    for q, docs in zip(questions, docs_per_question):
        top_titles = [d.get("title", "Sans titre") for d in docs[:3]]
        print(f"- Q: {q}")
        print(f"  Docs recuperes: {len(docs)}")
        if top_titles:
            print(f"  Top-3 titres: {top_titles}")


def print_metric_diagnostics(results_df: pd.DataFrame) -> None:
    metric_cols = [
        col
        for col in ["faithfulness", "context_utilization", "context_precision", "context_recall", "answer_relevancy"]
        if col in results_df.columns
    ]

    print("\n=== Scores moyens ===")
    means = results_df[metric_cols].mean(numeric_only=True)
    for metric in metric_cols:
        print(f"- {metric}: {means[metric]}")

    print("\n=== Questions les plus faibles par metrique ===")
    for metric in metric_cols:
        row = results_df.nsmallest(1, metric).iloc[0]
        print(f"- {metric}: {row[metric]:.4f} | question: {row['question']}")

    print("\n=== Lecture rapide des risques ===")
    for metric in metric_cols:
        score = means[metric]
        if pd.isna(score):
            print(f"- {metric}: NaN (metrique indisponible dans cette version de RAGAS)")
            continue
        if score < 0.60:
            level = "critique"
        elif score < 0.75:
            level = "a surveiller"
        else:
            level = "correct"
        print(f"- {metric}: {level} (score={score:.4f})")


def print_multirun_summary(all_scores: List[Dict[str, float]]) -> None:
    """Affiche moyenne et ecart-type sur plusieurs runs RAGAS."""
    if len(all_scores) < 2:
        return
    metrics = list(all_scores[0].keys())
    print("\n=== Synthese multi-runs ===")
    print(f"{'Metrique':<28} {'Moyenne':>8} {'Ecart-type':>12} {'Min':>8} {'Max':>8}")
    print("-" * 68)
    for metric in metrics:
        vals = [s[metric] for s in all_scores if not pd.isna(s.get(metric, float("nan")))]
        if not vals:
            print(f"{metric:<28} {'NaN':>8}")
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{metric:<28} {mean:>8.4f} {std:>12.4f} {min(vals):>8.4f} {max(vals):>8.4f}")


def main() -> None:
    args = parse_args()

    if args.env_file:
        env_path = Path(args.env_file)
        if not env_path.exists():
            raise FileNotFoundError(f"Fichier .env introuvable: {env_path}")
        load_dotenv(dotenv_path=env_path, override=True)

    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY manquant. Fournissez --env-file ou variable d'environnement.")

    nest_asyncio.apply()

    questions_test = build_test_questions()

    print("=== Initialisation chatbot RAG ===")
    chatbot = MistralRAGChatbot(index_dir="data", api_key=api_key, temperature=0.1)

    # Le RAG collection est identique a chaque run (on resample pour la variabilite LLM uniquement)
    answers, contexts, retrieved_docs = run_rag_collection(chatbot, questions_test, args.k)
    print_retrieval_overview(questions_test, retrieved_docs)

    print("\n=== Initialisation LLM/embeddings pour RAGAS ===")
    mistral_llm = ChatMistralAI(
        mistral_api_key=api_key,
        model="mistral-small-latest",
        temperature=0.0,
    )
    mistral_embeddings = MistralAIEmbeddings(mistral_api_key=api_key, model="mistral-embed")

    print("\n=== Generation de la ground_truth silver (alternative au labeling manuel) ===")
    ground_truths = [
        build_silver_ground_truth(mistral_llm, question=q, contexts=ctx)
        for q, ctx in zip(questions_test, contexts)
    ]

    evaluation_data = {
        "question": questions_test,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    evaluation_dataset = Dataset.from_dict(evaluation_data)

    metrics_to_evaluate = [
        faithfulness,
        # answer_relevancy,  # TODO: desactive en prod (retourne NaN avec RAGAS 0.1.22) — a reimplementer avec RAGAS >= 0.2.x
        context_utilization,  # fraction du contexte utilise dans la reponse — ne necessite PAS ground_truth
        context_precision,
        context_recall,
    ]
    print(f"\n=== Lancement evaluation RAGAS ({args.runs} run(s)) ===")
    print(f"Metriques: {[m.name for m in metrics_to_evaluate]}")

    all_run_scores: List[Dict[str, float]] = []

    for run_idx in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"\n--- Run {run_idx}/{args.runs} ---")
        results = evaluate(
            dataset=evaluation_dataset,
            metrics=metrics_to_evaluate,
            llm=mistral_llm,
            embeddings=mistral_embeddings,
        )

        results_df = results.to_pandas()

        print("\n=== Resultats bruts (par question) ===")
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        pd.set_option("display.max_colwidth", 120)
        display_cols = ["question", "faithfulness", "context_utilization", "context_precision", "context_recall"]
        # "answer_relevancy" exclu de l'affichage prod (colonne absente — metrique desactivee)
        print(results_df[display_cols])

        print_metric_diagnostics(results_df)

        # Enregistrer scores de ce run
        run_scores: Dict[str, float] = {}
        for metric in ["faithfulness", "context_utilization", "context_precision", "context_recall"]:
            if metric in results_df.columns:
                val = results_df[metric].mean(numeric_only=True)
                run_scores[metric] = float(val) if not pd.isna(val) else float("nan")
        all_run_scores.append(run_scores)

        if run_idx < args.runs:
            print(f"\n  (pause 30s avant run {run_idx + 1} pour eviter rate-limiting Mistral...)")
            import time
            time.sleep(30)

    print_multirun_summary(all_run_scores)

    print("\n=== Extraits de ground truth silver ===")
    for i, (q, gt) in enumerate(zip(questions_test, ground_truths), 1):
        one_line = " ".join(gt.split())
        print(f"- [{i}] {q}")
        print(f"  GT silver: {one_line[:220]}{'...' if len(one_line) > 220 else ''}")


if __name__ == "__main__":
    main()
