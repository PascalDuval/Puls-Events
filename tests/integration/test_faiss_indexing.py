"""
Tests du pipeline d'indexation FAISS.

Tests :
1. Création de l'index FAISS depuis vecteurs
2. Recherche sémantique basique
3. Recherche hybride (vecteur + filtres)
4. Performance et stats
"""

import os
import sys

import numpy as np
import pytest

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from faiss_searcher import FAISSSearcher


class TestFAISSIndexCreation:
    """Tests de création et chargement de l'index."""

    def test_index_exists(self):
        """Vérifier que les fichiers d'index existent."""
        assert os.path.exists("data/faiss_index.idx"), "Index FAISS manquant"
        assert os.path.exists(
            "data/faiss_metadata.pkl"
        ), "Métadonnées manquantes"

    def test_index_loading(self):
        """Charger l'index et vérifier les dimensions."""
        try:
            searcher = FAISSSearcher("data", verbose=False)
            assert searcher.index is not None
            assert searcher.index.ntotal > 0
            assert searcher.index.d == 1024  # Dimension Mistral
            assert len(searcher.metadata) == searcher.index.ntotal
        except Exception as e:
            pytest.fail(f"Erreur chargement index : {e}")

    def test_index_consistency(self):
        """Vérifier la cohérence entre index et métadonnées."""
        searcher = FAISSSearcher("data", verbose=False)
        # Chaque métadata doit avoir les champs essentiels
        for i, meta in enumerate(searcher.metadata):
            assert "id" in meta
            assert "title" in meta
            assert "event_start" in meta
            assert "city" in meta
            assert "tags" in meta
            assert isinstance(meta["tags"], list)


class TestSemanticSearch:
    """Tests de recherche sémantique."""

    @pytest.fixture
    def searcher(self):
        """Fixture pour créer un searcher."""
        return FAISSSearcher("data", verbose=False)

    def test_basic_search(self, searcher):
        """Tester une recherche basique."""
        # Créer un vecteur de requête aléatoire (normal mock)
        query = np.random.randn(1024).astype(np.float32).tolist()
        results = searcher.search(query, k=5)

        assert len(results) > 0
        assert len(results) <= 5
        assert all("rank" in r for r in results)
        assert all("distance" in r for r in results)
        assert all("similarity_score" in r for r in results)
        assert all(0 <= r["similarity_score"] <= 1 for r in results)

    def test_search_results_ranking(self, searcher):
        """Vérifier que les résultats sont bien rangés."""
        query = np.random.randn(1024).astype(np.float32).tolist()
        results = searcher.search(query, k=10)

        distances = [r["distance"] for r in results]
        # Vérifier que les distances sont croissantes (meilleur d'abord)
        assert distances == sorted(distances)

    def test_search_k_limit(self, searcher):
        """Tester que k est bien respecté."""
        query = np.random.randn(1024).astype(np.float32).tolist()

        for k in [1, 5, 10, 50]:
            results = searcher.search(query, k=k)
            assert len(results) <= k

    def test_search_returns_valid_metadata(self, searcher):
        """Vérifier que les résultats contiennent des métadonnées valides."""
        query = np.random.randn(1024).astype(np.float32).tolist()
        results = searcher.search(query, k=3)

        for res in results:
            assert isinstance(res["title"], str) and len(res["title"]) > 0
            assert isinstance(res["city"], str)
            assert isinstance(res["tags"], list)
            assert "source_record_url" in res or "source_url" in res


class TestHybridSearch:
    """Tests de recherche hybride (vecteur + filtres)."""

    @pytest.fixture
    def searcher(self):
        return FAISSSearcher("data", verbose=False)

    def test_hybrid_search_basic(self, searcher):
        """Tester la recherche hybride basique."""
        query = np.random.randn(1024).astype(np.float32).tolist()
        results = searcher.search_hybrid(query, k=5)

        assert isinstance(results, list)
        assert len(results) <= 5

    def test_hybrid_search_city_filter(self, searcher):
        """Tester le filtre par ville."""
        query = np.random.randn(1024).astype(np.float32).tolist()

        # Chercher les events à Paris
        results = searcher.search_hybrid(query, k=10, city="Paris")

        # Vérifier que tous les résultats sont à Paris (ou vides)
        if results:
            for res in results:
                assert res["city"].lower() == "paris"

    def test_hybrid_search_tags_filter(self, searcher):
        """Tester le filtre par tags."""
        query = np.random.randn(1024).astype(np.float32).tolist()

        # Chercher events avec tags "musique" ou "concert"
        results = searcher.search_hybrid(query, k=10, tags=["musique", "concert"])

        # Vérifier que tous les résultats ont au moins un des tags
        if results:
            filter_tags = {"musique", "concert"}
            for res in results:
                doc_tags = set(t.lower() for t in res.get("tags", []))
                assert doc_tags & filter_tags  # Au moins une intersection

    def test_hybrid_search_empty_results(self, searcher):
        """Tester que les filtres restrictifs peuvent retourner 0 résultats."""
        query = np.random.randn(1024).astype(np.float32).tolist()

        # Chercher une ville qui n'existe probablement pas
        results = searcher.search_hybrid(
            query, k=10, city="NonexistentCityXYZ"
        )

        assert len(results) == 0

    def test_hybrid_search_multiple_filters(self, searcher):
        """Tester plusieurs filtres combinés."""
        query = np.random.randn(1024).astype(np.float32).tolist()

        # Chercher events à Paris avec tag "exposition"
        results = searcher.search_hybrid(
            query, k=10, city="Paris", tags=["exposition"]
        )

        if results:
            for res in results:
                assert res["city"].lower() == "paris"
                doc_tags = {t.lower() for t in res.get("tags", [])}
                assert "exposition" in doc_tags


class TestIndexStats:
    """Tests des statistiques de l'index."""

    def test_get_stats(self):
        """Tester la récupération des stats."""
        searcher = FAISSSearcher("data", verbose=False)
        stats = searcher.get_stats()

        assert "total_documents" in stats
        assert "embedding_dimension" in stats
        assert "unique_cities" in stats
        assert "unique_regions" in stats
        assert "unique_tags" in stats

        assert stats["embedding_dimension"] == 1024
        assert stats["total_documents"] > 0
        assert stats["unique_cities"] > 0

    def test_get_by_id(self):
        """Tester la récupération par ID."""
        searcher = FAISSSearcher("data", verbose=False)

        if searcher.id_mapping:
            # Prendre un ID de test
            test_id = list(searcher.id_mapping.keys())[0]
            doc = searcher.get_by_id(test_id)

            assert doc is not None
            assert doc["id"] == test_id

    def test_nonexistent_id(self):
        """Tester la récupération d'un ID inexistant."""
        searcher = FAISSSearcher("data", verbose=False)
        doc = searcher.get_by_id("nonexistent-id-xyz-123")

        assert doc is None


class TestPerformance:
    """Tests de performance."""

    def test_search_speed(self):
        """Vérifier que les recherches sont rapides (< 100ms)."""
        import time

        searcher = FAISSSearcher("data", verbose=False)
        query = np.random.randn(1024).astype(np.float32).tolist()

        start = time.time()
        for _ in range(10):
            results = searcher.search(query, k=10)
        elapsed = time.time() - start
        avg_time_ms = (elapsed / 10) * 1000

        print(f"\n⏱️  Temps moyen par recherche : {avg_time_ms:.2f}ms")
        assert avg_time_ms < 100, f"Recherche trop lente : {avg_time_ms:.2f}ms"

    def test_index_memory_efficiency(self):
        """Vérifier la taille de l'index en mémoire."""
        index_size = os.path.getsize("data/faiss_index.idx") / 1024 / 1024
        print(f"\n💾 Taille index FAISS : {index_size:.2f} MB")

        # Index de 10k docs × 1024D en float32 = ~40 MB
        assert index_size < 500, f"Index trop gros : {index_size:.2f} MB"


if __name__ == "__main__":
    # Lancer les tests
    pytest.main([__file__, "-v", "-s"])
