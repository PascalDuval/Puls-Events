"""
Indexation vectorielle FAISS pour les événements culturels d'Ile-de-France.

Ce module charge les vecteurs Mistral depuis le fichier JSONL vectorisé,
crée un index FAISS optimisé pour la recherche sémantique rapide,
et stocke les métadonnées associées.

Utilisation :
    python index_events_faiss.py --api-key <CLE> [--input <path>] [--output-dir <path>]

Résultat :
    - data/faiss_index.idx : Index FAISS binaire (vectoriel)
    - data/faiss_metadata.pkl : Métadonnées compressées (titre, date, lieu, tags, URL)
    - data/faiss_id_mapping.pkl : Mapping docID -> metadonnées
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np


def load_vectorized_events(
    input_path: str, max_docs: Optional[int] = None
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str]]:
    """
    Charge les événements vectorisés depuis le fichier JSONL.

    Args:
        input_path: Chemin du fichier evenements_..._vectors.jsonl
        max_docs: Limite de documents à charger (test)

    Returns:
        (embeddings_array, metadata_list, id_list) :
            - embeddings_array: Array numpy (N, 1024) de vecteurs
            - metadata_list: Liste de dicts avec titre, date, ville, tags, etc.
            - id_list: Liste des IDs bruts
    """
    embeddings = []
    metadata = []
    doc_ids = []
    count = 0
    seen_ids = set()
    skipped_duplicates = 0

    print(f"[PHASE 1] Chargement des vecteurs depuis {input_path}...")
    start = time.time()

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if max_docs and count >= max_docs:
                    break

                try:
                    doc = json.loads(line.strip())
                except json.JSONDecodeError:
                    print(f"  ⚠ Ligne {count + 1} invalide, ignorée")
                    continue

                # Extraction du vecteur
                embedding = doc.get("embedding")
                if not embedding or not isinstance(embedding, list):
                    print(f"  ⚠ Doc {doc.get('id')} : embedding manquant")
                    continue

                if len(embedding) != 1024:
                    print(f"  ⚠ Doc {doc.get('id')} : dim={len(embedding)} (attendu 1024)")
                    continue

                doc_id = doc.get("id")
                if not doc_id:
                    print("  ⚠ Doc sans id ignore")
                    continue
                if doc_id in seen_ids:
                    skipped_duplicates += 1
                    continue
                seen_ids.add(doc_id)

                # Extraction des métadonnées
                meta = doc.get("metadata", {})
                meta_clean = {
                    "id": doc_id,
                    "title": meta.get("title", "N/A"),
                    "event_start": meta.get("event_start", ""),
                    "event_end": meta.get("event_end", ""),
                    "city": meta.get("city", ""),
                    "region": meta.get("region", ""),
                    "country": meta.get("country", "France"),
                    "tags": meta.get("tags", []),
                    "source_record_url": meta.get("source_record_url", ""),
                    "raw_uid": meta.get("raw_uid", ""),
                    "text_preview": doc.get("text", "")[:200] if doc.get("text") else "",
                }

                embeddings.append(embedding)
                metadata.append(meta_clean)
                doc_ids.append(doc_id)
                count += 1

                if count % 1000 == 0:
                    print(f"  [{count}] documents chargés...")

    except FileNotFoundError:
        print(f"❌ Fichier non trouvé : {input_path}")
        sys.exit(1)

    elapsed = time.time() - start
    print(
        f"[PHASE 1] Terminé en {elapsed:.1f}s — "
        f"chargés={count}, dim=1024, taille_array=({count}, 1024)"
    )
    if skipped_duplicates:
        print(f"[PHASE 1] Doublons ignores pendant le chargement: {skipped_duplicates}")

    # Conversion numpy
    embeddings_array = np.array(embeddings, dtype=np.float32)
    return embeddings_array, metadata, doc_ids


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Crée un index FAISS optimisé pour la recherche sémantique.

    Stratégie :
    - Utilise IndexFlatL2 (recherche exacte par distance euclidienne L2)
    - Robuste, pas de perte, performance excellente pour ~10k documents
    - Dimension : 1024 (sortie de mistral-embed)

    Pour des datasets plus gros (> 100k docs), on pourrait utiliser :
    - IndexIVFFlat : indexation par clustering (10x plus rapide)
    - IndexHNSW : recherche hiérarchique (très rapide)

    Args:
        embeddings: Array (N, 1024) de vecteurs float32

    Returns:
        Index FAISS prêt pour la recherche
    """
    dim = embeddings.shape[1]
    n_docs = embeddings.shape[0]

    print(f"\n[PHASE 2] Construction de l'index FAISS...")
    print(f"  Dimensions : {n_docs} documents × {dim} dimensions")
    print(f"  Stratégie : IndexFlatL2 (recherche exacte)")

    start = time.time()

    # Créer l'index : distance L2 (euclidienne)
    index = faiss.IndexFlatL2(dim)

    # Ajouter les vecteurs
    index.add(embeddings)

    elapsed = time.time() - start

    print(f"  Index créé en {elapsed:.2f}s")
    print(f"  ✓ {index.ntotal} vecteurs indexés")

    return index


def save_index_and_metadata(
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    doc_ids: List[str],
    output_dir: str,
) -> None:
    """
    Sauvegarde l'index FAISS et les métadonnées sur disque.

    Fichiers générés :
    - faiss_index.idx : Index binaire FAISS (pas de métadonnées)
    - faiss_metadata.pkl : Métadonnées compressées (titre, date, ville, tags, URL)
    - faiss_id_mapping.pkl : Mapping docID -> index pour retrouver facilement

    Args:
        index: Index FAISS compilé
        metadata: Liste des métadonnées (position i = doc i)
        doc_ids: Liste des IDs bruts
        output_dir: Dossier de destination
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n[PHASE 3] Sauvegarde sur disque...")

    # Sauvegarder l'index FAISS
    index_path = os.path.join(output_dir, "faiss_index.idx")
    faiss.write_index(index, index_path)
    index_size_mb = os.path.getsize(index_path) / 1024 / 1024
    print(f"  ✓ Index FAISS : {index_path} ({index_size_mb:.2f} MB)")

    # Sauvegarder les métadonnées
    metadata_path = os.path.join(output_dir, "faiss_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    metadata_size_mb = os.path.getsize(metadata_path) / 1024 / 1024
    print(f"  ✓ Métadonnées : {metadata_path} ({metadata_size_mb:.2f} MB)")

    # Sauvegarder le mapping ID -> index
    id_mapping = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    id_mapping_path = os.path.join(output_dir, "faiss_id_mapping.pkl")
    with open(id_mapping_path, "wb") as f:
        pickle.dump(id_mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  ✓ ID mapping : {id_mapping_path}")


def semantic_search(
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    query_text: str,
    query_embedding: List[float],
    k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Effectue une recherche sémantique dans l'index FAISS.

    Args:
        index: Index FAISS
        metadata: Liste des métadonnées (position i = doc i)
        query_text: Texte de la requête (pour affichage)
        query_embedding: Vecteur de requête (1024D)
        k: Nombre de résultats à retourner

    Returns:
        Liste de dicts {distance, title, city, tags, source_url, text_preview}
    """
    query_vec = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query_vec, k)

    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < 0 or idx >= len(metadata):
            continue

        meta = metadata[idx]
        results.append({
            "rank": i + 1,
            "distance": float(dist),
            "similarity_score": 1.0 / (1.0 + float(dist)),  # Conversion L2 -> similarité [0,1]
            "title": meta.get("title"),
            "city": meta.get("city"),
            "region": meta.get("region"),
            "tags": meta.get("tags"),
            "event_start": meta.get("event_start"),
            "text_preview": meta.get("text_preview"),
            "source_url": meta.get("source_record_url"),
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Indexation FAISS pour recherche sémantique d'événements culturels d'Ile-de-France"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/evenements_publics_openagenda_culture_ile_de_france_vectors.jsonl",
        help="Chemin du fichier vectorisé JSONL (défaut: data/...vectors.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Dossier de sortie pour index et métadonnées (défaut: data/)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limite de docs à indexer (test)",
    )
    parser.add_argument(
        "--test-query",
        type=str,
        default=None,
        help="Requête de test pour valider la recherche (ex: 'festival musique Paris')",
    )

    args = parser.parse_args()

    # ========== PHASE 1 : Charger les vecteurs ==========
    embeddings, metadata, doc_ids = load_vectorized_events(
        args.input, max_docs=args.max_docs
    )

    # ========== PHASE 2 : Créer l'index FAISS ==========
    index = create_faiss_index(embeddings)

    # ========== PHASE 3 : Sauvegarder ==========
    save_index_and_metadata(index, metadata, doc_ids, args.output_dir)

    print(f"\n[DONE] Index FAISS créé avec succès !")
    print(f"  Total documents indexés : {index.ntotal}")
    print(f"  Dimension embeddings : 1024")
    print(f"  Algorithme : IndexFlatL2 (recherche exacte)")

    # ========== TEST (optionnel) ==========
    if args.test_query:
        print(f"\n[TEST] Requête de test : '{args.test_query}'")
        # Pour ce test, utiliser une requête fictive (mock embedding)
        # En production, il faudrait utiliser mistral-embed pour vectoriser la requête
        test_embedding = np.random.randn(1024).astype(np.float32).tolist()
        results = semantic_search(
            index, metadata, args.test_query, test_embedding, k=3
        )
        print(f"  Résultats (top-3) :")
        for res in results:
            print(
                f"    [{res['rank']}] {res['title']} ({res['city']}) | "
                f"score={res['similarity_score']:.3f}"
            )


if __name__ == "__main__":
    main()
