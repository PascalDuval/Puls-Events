"""
Démonstration du pipeline RAG complet : recherche sémantique avec FAISS.

Usage :
    python tools/secondary/demo_faiss_search.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faiss_searcher import FAISSSearcher


def demo_basic_search() -> None:
    print("=" * 70)
    print("DEMO 1 : Recherche semantique par similarite")
    print("=" * 70)

    searcher = FAISSSearcher(str(PROJECT_ROOT / "data"))
    query = np.random.randn(1024).astype(np.float32).tolist()
    results = searcher.search(query, k=5)

    print(f"\nTrouve {len(results)} resultats:")
    for res in results:
        print(f"\n  [{res['rank']}] {res['title']}")
        print(f"      Ville : {res['city']} ({res['region']})")
        print(f"      Date : {res['event_start'][:10]}")


def demo_hybrid_search_city() -> None:
    print("\n\n" + "=" * 70)
    print("DEMO 2 : Recherche hybride (vecteur + filtres)")
    print("=" * 70)

    searcher = FAISSSearcher(str(PROJECT_ROOT / "data"))
    query = np.random.randn(1024).astype(np.float32).tolist()
    results = searcher.search_hybrid(query, k=5, city="Paris", tags=["exposition", "musée"])

    print(f"\nTrouve {len(results)} resultats a Paris avec exposition/musee:")
    for res in results:
        print(f"  - {res['title']} ({res['event_start'][:10]})")


def main() -> None:
    demo_basic_search()
    demo_hybrid_search_city()


if __name__ == "__main__":
    main()
