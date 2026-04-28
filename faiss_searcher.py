"""
Module d'interaction avec l'index FAISS pour recherche sémantique.

Fournit une classe FAISSSearcher pour :
- Charger l'index FAISS et les métadonnées
- Effectuer des recherches sémantiques
- Filtrer par métadonnées (ville, date, tags)
- Renvoyer des résultats formatés

Utilisation :
    from faiss_searcher import FAISSSearcher
    
    searcher = FAISSSearcher("data")
    results = searcher.search_hybrid(query_embedding, city="Paris", k=5)
"""

from __future__ import annotations

import os
import pickle
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import faiss
import numpy as np


class FAISSSearcher:
    """
    Wrapper pour recherche sémantique avec index FAISS.

    Supports :
    - Recherche par similarité vectorielle
    - Filtrage par métadonnées (ville, région, tags, date)
    - Recherche hybride (vecteur + filtres)
    """

    def __init__(self, index_dir: str = "data", verbose: bool = True):
        """
        Initialise le searcher en chargeant l'index et les métadonnées.

        Args:
            index_dir: Dossier contenant faiss_index.idx et faiss_metadata.pkl
            verbose: Affiche les logs
        """
        self.index_dir = index_dir
        self.verbose = verbose
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.id_mapping: Dict[str, int] = {}

        self._load()

    def _load(self) -> None:
        """Charge l'index FAISS et les métadonnées depuis le disque."""
        # Index FAISS
        index_path = os.path.join(self.index_dir, "faiss_index.idx")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index FAISS non trouvé : {index_path}")

        self.index = faiss.read_index(index_path)
        if self.verbose:
            print(f"✓ Index FAISS chargé : {self.index.ntotal} vecteurs")

        # Métadonnées
        metadata_path = os.path.join(self.index_dir, "faiss_metadata.pkl")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Métadonnées non trouvées : {metadata_path}")

        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        if self.verbose:
            print(f"✓ Métadonnées chargées : {len(self.metadata)} documents")

        # ID mapping (optionnel)
        id_mapping_path = os.path.join(self.index_dir, "faiss_id_mapping.pkl")
        if os.path.exists(id_mapping_path):
            with open(id_mapping_path, "rb") as f:
                self.id_mapping = pickle.load(f)

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Recherche sémantique pure (par vecteur).

        Args:
            query_embedding: Vecteur de requête (1024D)
            k: Nombre de résultats

        Returns:
            Liste des top-k résultats avec metadata
        """
        if not self.index:
            raise RuntimeError("Index FAISS non chargé")

        query_vec = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vec, k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if 0 <= idx < len(self.metadata):
                meta = self.metadata[idx].copy()
                meta.update({
                    "rank": rank + 1,
                    "distance": float(dist),
                    "similarity_score": self._l2_to_similarity(float(dist)),
                })
                results.append(meta)

        return results

    def search_hybrid(
        self,
        query_embedding: List[float],
        k: int = 5,
        city: Optional[str] = None,
        region: Optional[str] = None,
        tags: Optional[List[str]] = None,
        after_date: Optional[str] = None,
        before_date: Optional[str] = None,
        intent_tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recherche hybride : vecteur + filtres métadonnées + reranking léger.

        Stratégie :
        1. Recherche vectorielle top-max(k*5, 50)
        2. Filtrage par métadonnées (ville, région, tags, dates)
        3. Reranking par score combiné : similarité + bonus tags + bonus ville
        4. Retour des k meilleurs après reranking

        Args:
            query_embedding: Vecteur de requête (1024D)
            k: Nombre de résultats finaux
            city: Filtrer par ville (exact)
            region: Filtrer par région (exact)
            tags: Filtrer par tags (au moins un doit matcher)
            after_date: Filtrer events après cette date (ISO 8601)
            before_date: Filtrer events avant cette date (ISO 8601)
            intent_tags: Tags d'intention (pour reranking, sans filtre dur)

        Returns:
            Liste des résultats filtrés et rerankés
        """
        # Recherche vectorielle sur plus de candidats pour le reranking
        top_k_search = max(k * 5, 50)
        candidates = self.search(query_embedding, k=top_k_search)

        # Appliquer les filtres métadonnées
        filtered = []
        for doc in candidates:
            if city and doc.get("city", "").lower() != city.lower():
                continue
            if region and doc.get("region", "").lower() != region.lower():
                continue
            if tags:
                doc_tags = set(self._normalize_tag(t) for t in doc.get("tags", []))
                filter_tags = set(self._normalize_tag(t) for t in tags)
                doc_tags.discard("")
                filter_tags.discard("")
                if not doc_tags & filter_tags:  # Pas d'intersection
                    continue
            if after_date:
                # Chevauchement : l'evenement doit ne pas etre termine avant la fenetre
                # (event_end >= after_date). Si event_end absent, on se rabat sur event_start.
                event_end = doc.get("event_end", "") or doc.get("event_start", "")
                if event_end and event_end < after_date:
                    continue
            if before_date:
                # L'evenement doit avoir commence avant la fin de la fenetre
                event_start = doc.get("event_start", "")
                if event_start and event_start > before_date:
                    continue

            filtered.append(doc)

        # Reranking léger : similarité de base + bonus sémantiques
        rerank_tags = set(self._normalize_tag(t) for t in (intent_tags or tags or []))
        rerank_tags.discard("")

        for doc in filtered:
            score = doc.get("similarity_score", 0.0)

            # Bonus tags : +0.15 si ≥2 tags matchent, +0.08 si 1 seul
            if rerank_tags:
                doc_tags = set(self._normalize_tag(t) for t in doc.get("tags", []))
                doc_tags.discard("")
                matched = len(doc_tags & rerank_tags)
                if matched >= 2:
                    score += 0.15
                elif matched == 1:
                    score += 0.08

            # Bonus ville : +0.10 si ville exacte présente dans les filtres
            if city and doc.get("city", "").lower() == city.lower():
                score += 0.10

            doc["rerank_score"] = score

        # Trier par rerank_score décroissant
        filtered.sort(key=lambda d: d.get("rerank_score", 0.0), reverse=True)

        return filtered[:k]

    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Récupère une document par son ID."""
        if doc_id not in self.id_mapping:
            return None
        idx = self.id_mapping[doc_id]
        if 0 <= idx < len(self.metadata):
            return self.metadata[idx]
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Retourne des statistiques sur l'index."""
        if not self.index or not self.metadata:
            return {}

        cities = set(m.get("city", "?") for m in self.metadata if m.get("city"))
        regions = set(m.get("region", "?") for m in self.metadata if m.get("region"))
        all_tags = set()
        for m in self.metadata:
            all_tags.update(m.get("tags", []))

        return {
            "total_documents": self.index.ntotal,
            "embedding_dimension": self.index.d,
            "index_type": str(type(self.index).__name__),
            "unique_cities": len(cities),
            "unique_regions": len(regions),
            "unique_tags": len(all_tags),
            "sample_tags": sorted(list(all_tags))[:20],
        }

    @staticmethod
    def _l2_to_similarity(l2_distance: float) -> float:
        """Convertit distance L2 en score de similarité [0, 1]."""
        # Formule : similarity = 1 / (1 + distance)
        # distance=0 -> similarity=1.0, distance=∞ -> similarity=0
        return 1.0 / (1.0 + l2_distance)

    @staticmethod
    def _normalize_tag(tag: Any) -> str:
        """Normalise un tag pour les comparaisons (casse, # et guillemets)."""
        text = str(tag or "").strip().lower()
        text = text.strip("\"'`“”‘’«»")
        text = re.sub(r"^#+", "", text)
        return text.strip()


if __name__ == "__main__":
    # Test simple
    import sys

    try:
        searcher = FAISSSearcher("data", verbose=True)

        stats = searcher.get_stats()
        print("\n📊 Statistiques index :")
        for key, val in stats.items():
            if key == "sample_tags":
                print(f"  {key}: {val[:5]}...")
            else:
                print(f"  {key}: {val}")

        print("\n✓ FAISSSearcher prêt pour recherche sémantique")

    except Exception as e:
        print(f"❌ Erreur : {e}", file=sys.stderr)
        sys.exit(1)
