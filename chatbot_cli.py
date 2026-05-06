from __future__ import annotations

import argparse
import json
from typing import List, Optional

from rag_chatbot_mistral import MistralRAGChatbot
from utils.temporal_deixis import infer_temporal_window


def _split_csv(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [p.strip() for p in value.split(",") if p.strip()] or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chatbot RAG evenements (FAISS + mistral-small-latest)."
    )
    parser.add_argument("--question", required=True, help="Question utilisateur")
    parser.add_argument("--k", type=int, default=6, help="Top-K retrieval")
    parser.add_argument("--city", type=str, default=None)
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None, help="Liste CSV de tags (override auto-inference)")
    parser.add_argument("--after-date", type=str, default=None, help="Filtre ISO 8601 (override auto-detection)")
    parser.add_argument("--before-date", type=str, default=None, help="Filtre ISO 8601 (override auto-detection)")
    parser.add_argument("--json", action="store_true", help="Sortie JSON")
    parser.add_argument("--verbose", action="store_true", help="Affiche les filtres inferres et les docs recuperes")
    args = parser.parse_args()

    # Apercu verbose : afficher la fenetre temporelle qui sera auto-detectee
    if args.verbose and not args.after_date and not args.before_date:
        tw = infer_temporal_window(args.question)
        if tw:
            print(f"[verbose] Fenetre temporelle auto-detectee : {tw.label} ({tw.after_date_utc} -> {tw.before_date_utc})")
        else:
            print("[verbose] Aucune fenetre temporelle detectee dans la question.")

    try:
        bot = MistralRAGChatbot()
        result = bot.ask(
            question=args.question,
            k=args.k,
            city=args.city,
            region=args.region,
            tags=_split_csv(args.tags),
            after_date=args.after_date,
            before_date=args.before_date,
        )
    except Exception as exc:
        print(f"Erreur: {exc}")
        return

    if args.verbose:
        print(f"[verbose] Documents recuperes : {len(result.documents)}")
        for i, doc in enumerate(result.documents, 1):
            title = doc.get("title", "Sans titre")
            score = doc.get("rerank_score", doc.get("similarity_score", "?"))
            tags_preview = ", ".join(doc.get("tags", [])[:4])
            print(f"  [{i}] {title} | score={score:.4f} | tags={tags_preview}")

    if args.json:
        print(
            json.dumps(
                {
                    "answer": result.answer,
                    "sources": result.sources,
                    "documents_count": len(result.documents),
                    "model": result.model,
                    "documents": [
                        {
                            "title": d.get("title", ""),
                            "city": d.get("city", ""),
                            "rerank_score": d.get("rerank_score", d.get("similarity_score")),
                            "tags": d.get("tags", [])[:6],
                            "url": d.get("source_record_url", ""),
                        }
                        for d in result.documents
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(result.answer)


if __name__ == "__main__":
    main()
