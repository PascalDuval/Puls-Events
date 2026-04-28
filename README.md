# Pull-Events - OpenAgenda IDF et pipeline RAG culturel

## Presentation

Ce dossier contient un pipeline RAG complet pour recommander des evenements culturels en Ile-de-France a partir de donnees OpenAgenda, d'embeddings Mistral et d'un index FAISS.

La fenetre temporelle de ces évenements temporels est :
- debut de fenetre : 25/04/2025 00:00:00 UTC
- fin de fenetre : 25/04/2027 23:59:59 UTC



Le projet couvre toute la chaine technique suivante :

1. collecte OpenAgenda,
2. nettoyage et normalisation des enregistrements,
3. generation d'un corpus JSONL RAG,
4. vectorisation des documents,
5. construction et validation d'un index FAISS,
6. recherche hybride vecteur + metadonnees,
7. reponse chatbot guidee par les documents recuperes.



## Resume executif

Etat de la base d'artefacts versionnee au 27/04/2026 :

- Documents RAG : 8014
- Lignes du fichier de vecteurs : 8014
- IDs uniques dans les vecteurs : 8014
- Taux d'alignement RAG -> vecteurs : 100%
- Vecteurs indexes dans FAISS : 8014
- Dimension embedding : 1024
- Tests pipeline : 41/41 PASS
- Taille totale des artefacts d'index : 36.00 MB

Validation de non-regression effectuee (sur corpus regenere 8014 docs) :

- `tests/unit`: 23/23 PASS
- `tests/integration/test_rag_quality_guard.py`: 1/1 PASS
- `tests/integration/test_faiss_indexing.py`: 17/17 PASS

## Journal du pipeline

### Etape 1 - Localisation du point de controle

Le point de controle de la fenetre temporelle se trouve dans `openagenda_culture_france_rag.py` via deux constantes :

- `WINDOW_START_UTC`
- `WINDOW_END_UTC`

### Etape 2 - Correction de la fenetre

La correction appliquee dans le code source est minimale et ciblee :

- borne basse passee de `2026-04-25` a `2025-04-25`,
- borne haute conservee a `2027-04-25`,
- commentaire de garde d'integration aligne sur la meme fenetre.

### Etape 3 - Validation ciblee immediate

Verification executee juste apres la modification de code :

- `pytest tests/integration/test_rag_quality_guard.py -q`
- resultat : `1 passed`

### Etape 4 - Verification pipeline complete

La suite de tests effectivement utilisee comme pipeline a ensuite ete rejouee :

- `pytest tests/unit -q`
- `pytest tests/integration/test_rag_quality_guard.py -q`
- `pytest tests/integration/test_faiss_indexing.py -q`

Resultat consolide :

- 38 tests passes sur 38
- 3 warnings de depreciation SWIG observables sur les suites qui manipulent FAISS
- warnings non bloquants et sans impact fonctionnel sur la recherche ou l'indexation

### Etape 5 - Relecture des artefacts et mise a jour de la documentation

Les artefacts versionnes ont ete reanalyses pour produire un README complet contenant :

- procedure de reproduction,
- architecture,
- KPI,
- validation des index,
- benchmark FAISS,
- bilan des reponses chatbot,
- proposition d'amelioration de la recherche temporelle sans implementation immediate.

## Gestion des dependances

Deux options sont documentees ci-dessous. L'option recommandee sur ce workspace reste conda.

### Option recommandee : conda

Interpreteur recommande pour ce projet :

```bash
conda activate LLMRag
```

Puis installation des dependances Python du projet :

```bash
pip install -r Pull-Events/requirements.txt
```

### Option alternative : pip pur

Si vous utilisez deja un environnement virtuel actif :

```bash
python -m pip install --upgrade pip
python -m pip install -r Pull-Events/requirements.txt
```

### Dependances critiques du pipeline

Les composants les plus importants sont :

- `requests` pour la collecte OpenAgenda,
- `python-dateutil` pour le parsing robuste des dates,
- `mistralai` pour les embeddings et la generation,
- `faiss-cpu` pour l'index vectoriel,
- `numpy` pour la manipulation des embeddings,
- `pytest` pour la validation.

## Instructions de reproduction

## Procedure inversee (depuis le depot distant)

Cette procedure permet a un tout nouvel utilisateur de repartir de zero depuis GitHub, d'installer l'environnement puis d'executer le pipeline dans le bon ordre.

### 1. Cloner le depot distant

```bash
git clone https://github.com/PascalDuval/Puls-Events.git
cd Puls-Events/Pull-Events
```

### 2. Creer et activer l'environnement

Option conda (recommandee) :

```bash
conda create -n LLMRag python=3.10 -y
conda activate LLMRag
pip install -r requirements.txt
```

Option venv :

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

### 3. Definir la cle API Mistral

Configurer `MISTRAL_API_KEY` dans l'environnement (ou via un fichier `.env` compatible avec vos scripts).

### 4. Lancer les scripts dans l'ordre (adaptation possible)

1. `openagenda_culture_france_rag.py` : collecte OpenAgenda + generation du corpus RAG JSONL.
2. `vectorize_events_mistral.py` : creation des embeddings Mistral a partir du corpus.
3. `index_events_faiss.py` : construction de l'index FAISS + metadonnees associees.
4. `chatbot_cli.py` (optionnel) : validation rapide en ligne de commande.
5. `PullEventsIDFBot.py` via Streamlit : interface utilisateur finale.

Exemples de lancement :

```bash
python openagenda_culture_france_rag.py
python vectorize_events_mistral.py
python index_events_faiss.py
python chatbot_cli.py --question "as-tu un concert de jazz a Paris ?"
python -m streamlit run PullEventsIDFBot.py
```

### 5. Choix rapide : lancer directement Streamlit

Si les artefacts (`data/evenements_..._rag.jsonl`, `data/evenements_..._vectors.jsonl`, `data/faiss_index.idx` et fichiers metadata) sont deja presents et coherents, vous pouvez lancer directement :

```bash
python -m streamlit run PullEventsIDFBot.py
```

Les commandes ci-dessous sont celles a executer depuis le dossier `Pull-Events/`.

### 1. Activer l'environnement

```bash
conda activate LLMRag
```

### 2. Installer les dependances

```bash
pip install -r requirements.txt
```

### 3. (Re)generer le corpus RAG OpenAgenda

```bash
C:/Users/karap/anaconda3/envs/LLMRag/python.exe openagenda_culture_france_rag.py
```

Sortie attendue :

- `data/evenements_publics_openagenda_culture_ile_de_france_rag.jsonl`

### 4. Vectoriser le corpus

Si vous disposez d'un fichier `.env` contenant `MISTRAL_API_KEY` :

```bash
C:/Users/karap/anaconda3/envs/LLMRag/python.exe vectorize_events_mistral.py --env-file "C:/Users/karap/OpenClassRooms/projet11/coursEtExos/8532116-mettez-en-place-un-rag-pour-un-llm/.env"
```

Sinon, vous pouvez fournir la cle directement par variable d'environnement ou argument CLI selon votre mode d'execution.

Sortie attendue :

- `data/evenements_publics_openagenda_culture_ile_de_france_vectors.jsonl`

### 5. Construire l'index FAISS

```bash
C:/Users/karap/anaconda3/envs/LLMRag/python.exe index_events_faiss.py
```

Artefacts generes :

- `data/faiss_index.idx`
- `data/faiss_metadata.pkl`
- `data/faiss_id_mapping.pkl`

### 6. Executer les tests

```bash
C:/Users/karap/anaconda3/envs/LLMRag/python.exe -m pytest tests/unit -q
C:/Users/karap/anaconda3/envs/LLMRag/python.exe -m pytest tests/integration/test_rag_quality_guard.py -q
C:/Users/karap/anaconda3/envs/LLMRag/python.exe -m pytest tests/integration/test_faiss_indexing.py -q
```

### 7. Interroger le chatbot

```bash
C:/Users/karap/anaconda3/envs/LLMRag/python.exe chatbot_cli.py --question "as-tu un concert de jazz a Paris ?"
```

### 8. Lancer l'interface Streamlit

```bash
cd Pull-Events
C:/Users/karap/anaconda3/envs/LLMRag/python.exe -m streamlit run PullEventsIDFBot.py
```

L'interface s'ouvre automatiquement dans le navigateur a l'adresse `http://localhost:8501`.

### 9. Verifier manuellement les statistiques d'index

```bash
C:/Users/karap/anaconda3/envs/LLMRag/python.exe -m pytest tests/integration/test_faiss_indexing.py -q
```

## Description des fichiers et dossiers

### Fichiers principaux a la racine

- `openagenda_culture_france_rag.py` : collecte OpenAgenda, filtrage IDF, fenetre temporelle fixe, normalisation JSONL RAG.
- `vectorize_events_mistral.py` : preparation de texte, gestion du fichier existant, generation des embeddings Mistral.
- `index_events_faiss.py` : chargement des embeddings, construction de l'index, sauvegarde des metadonnees et de l'ID mapping.
- `faiss_searcher.py` : recherche semantique pure et recherche hybride avec filtres metadata.
- `rag_chatbot_mistral.py` : orchestration embed -> retrieve -> generate.
- `chatbot_cli.py` : point d'entree CLI pour interroger le chatbot.
- `PullEventsIDFBot.py` : interface conversationnelle Streamlit (voir section [Interface Streamlit](#interface-streamlit--lancement-fonctionnement-et-panneau-de-fiabilite)).
- `README.md` : documentation technique et guide de reproduction.

### Dossier `data/`

Contient les artefacts versionnes de reference :

- corpus RAG JSONL,
- corpus vectorise JSONL,
- index FAISS,
- metadonnees pickle,
- mapping ID pickle,
- quelques illustrations et supports visuels.

### Dossier `tests/`

- `tests/unit/` : tests unitaires de scripts et helpers.
- `tests/integration/` : tests sur la qualite du JSONL et le chargement / comportement de l'index FAISS.
- `tests/manual/` : scripts de verification manuelle et tests non automatises.
- `tests/conftest.py` : configuration Pytest et stabilisation des imports.

### Dossier `tools/`

- `tools/diagnostic/` : scripts de verification ponctuelle de l'API, des filtres et du comportement de collecte.
- `tools/secondary/` : scripts utilitaires de demonstration et d'inspection (dont `temp_demo.py` : affichage d'un evenement vectorise exemple depuis le corpus).

### Dossier `utils/`

- `utils/temporal_deixis.py` : parsing des expressions temporelles deictiques francaises (`ce soir`, `demain`, `ce week-end`, `en mai`, etc.) — retourne une `TemporalWindow` UTC utilisee par le chatbot.

### Dossier `old/`

Historique technique du projet : artefacts d'essais, extractions temporaires, notes et anciens diagnostics. Ce dossier ne doit pas servir de reference principale pour la reproduction courante.

## Architecture du pipeline

Le pipeline complet suit le flux suivant :

```text
OpenAgenda -> JSONL RAG -> embeddings Mistral -> index FAISS -> recherche hybride -> generation de reponse
```

Traduction en scripts :

```text
openagenda_culture_france_rag.py
  -> vectorize_events_mistral.py
  -> index_events_faiss.py
  -> faiss_searcher.py
  -> rag_chatbot_mistral.py
  -> chatbot_cli.py
```

## Processus de nettoyage des donnees

Le nettoyage est un point central du projet, car la qualite des donnees conditionne directement la qualite des embeddings, la precision de la recherche et la fiabilite des reponses.

### Nettoyage applique dans la collecte

Le script `openagenda_culture_france_rag.py` met en place plusieurs niveaux de nettoyage :

1. normalisation des espaces avec `clean_text`,
2. suppression du HTML avec `strip_html`,
3. parsing robuste des dates avec `parse_dt`,
4. reconstitution de fenetres d'evenement a partir de `timings` avec `parse_timings_window`,
5. reparation d'intervalles dates incoherents avec `sanitize_date_range`,
6. deduplication des tags en preservant l'ordre,
7. verification de la vectorisabilite avec `is_vectorizable`,
8. verification des URLs source HTTP/HTTPS,
9. deduplication des documents par `raw_uid` ou cle de secours.

### Nettoyage applique avant vectorisation

Le script `vectorize_events_mistral.py` ajoute un second filet de securite :

1. reconstruction d'un texte de secours si `content` est trop pauvre,
2. normalisation du texte d'entree pour l'embedding,
3. filtrage des documents trop courts,
4. validation stricte de la presence d'une URL source,
5. gestion propre du fichier vectorise existant,
6. compactage de sortie pour supprimer les doublons ou lignes non pertinentes.

### Resultat qualite observable

Sur les 8014 documents RAG versionnes :

- 3 documents seulement remontent un champ manquant dans `metadata.quality_missing_fields`,
- ces 3 cas concernent uniquement le champ `city`,
- aucun document n'est invalide pour la vectorisation dans la suite de validation integration.

## Scripts de pre-processing et documentation integree par docstrings

Tous les scripts du pipeline sont documentes via des **docstrings de style Google/NumPy**
couvrant chaque module (description, entrees, sorties, usage) et chaque fonction publique
(Args / Returns / Raises). Cette documentation integree garantit que les hypotheses
de conception restent visibles a la maintenance et que les artefacts produits
sont explicitement nommes.

### Scripts de pre-processing (collecte et preparation du corpus)

- **`openagenda_culture_france_rag.py`** — docstring module + toutes les fonctions :
  `parse_dt`, `first_non_empty`, `to_text`, `clean_text`, `strip_html`,
  `parse_timings_window`, `sanitize_date_range`, `extract_tags`, `is_vectorizable`,
  `looks_cultural`, `is_ile_de_france`, `extract_dates`, `overlaps_window`,
  `build_window_where`, `normalize_record`.

- **`vectorize_events_mistral.py`** — docstring module + toutes les fonctions :
  `clean_text`, `first_non_empty`, `to_tag_list`, `build_fallback_text`,
  `normalize_source_url`, `build_doc_id`, `prepare_embedding_document`, `chunked`,
  `embed_texts`, `_load_already_written_ids`, `_compact_output_file`.

### Scripts d'indexation et recherche

- **`index_events_faiss.py`** — docstring module + fonctions :
  `load_vectorized_events`, `create_faiss_index`, `save_index_and_metadata`, `semantic_search`.

- **`faiss_searcher.py`** — docstring module, classe `FAISSSearcher` et methodes :
  `search`, `search_hybrid`, `get_by_id`, `get_stats`.

- **`rag_chatbot_mistral.py`** — docstrings sur `RAGChatbot` et son cycle
  `embed -> retrieve -> generate`, guardrails, et helpers internes.

## Scripts de vectorisation et creation / gestion d'index vectoriels

### Vectorisation

Le script `vectorize_events_mistral.py` :

- lit le JSONL RAG,
- prepare le texte a embedder,
- appelle `mistral-embed`,
- conserve les metadonnees utiles pour la recherche hybride,
- ecrit `evenements_publics_openagenda_culture_ile_de_france_vectors.jsonl`.

Chaque ligne du fichier vectorise contient :

- un `id`,
- un `text`,
- un `embedding`,
- un bloc `metadata` contenant au minimum titre, dates, ville, region, tags, pays et URL source.

#### Exemple concret d'événement vectorisé

**Événement :** `Les expositions` à **Paris** (2025-09-23)

**Métadonnées :**
- Tags : Sciences de la vie, Habitat - territoire - transport, Environnement - developpement durable - energies, Mathematiques - physique - chimie, Innovation - recherche - industrie - ingenierie
- URL : https://openagenda.com/echosciences/events/explora_507

**Vecteur embedding (1024 dimensions, premiers 64 éléments) :**

```python
[-0.043701, 0.029251, 0.068359, 0.004639, 0.008179, 0.004433, 0.033081, 0.009323,
  0.008911, 0.001840, -0.037659, 0.037659, 0.020203, -0.009140, -0.050446, -0.002901,
  0.003975, 0.030533, 0.017090, 0.003199, -0.024490, -0.000554, -0.021851, -0.023941,
  0.026505, 0.013069, -0.036926, -0.050812, -0.031616, 0.003975, 0.021027, -0.016449,
  0.000383, -0.010788, 0.003084, -0.002308, 0.015717, -0.056671, 0.002399, -0.026505,
  -0.006351, -0.004639, 0.010971, -0.034180, 0.024307, -0.032715, 0.023941, -0.012062,
  0.008087, -0.033997, 0.010193, 0.033447, -0.004593, -0.008362, -0.027420, 0.050812,
  0.022034, 0.011429, -0.039307, 0.026505, -0.048615, -0.016083, -0.037292, 0.006626,
  ... (960 dimensions supplémentaires) ...]
```

**Statistiques du vecteur :**
- **Dimension** : 1024
- **Min** : -0.103088 | **Max** : 0.091064
- **Moyenne** : 0.000319 (proche de zéro = normalisation)
- **Norme L2** : 1.000008 (vecteur normalisé pour le calcul de distance)

Chacun des 1024 nombres capture un aspect sémantique différent du titre et du contexte. La **norme L2 = 1** permet à FAISS de calculer efficacement les distances L2 pour retrouver les événements **sémantiquement proches**.

### Creation d'index

Le script `index_events_faiss.py` :

- charge les embeddings en `float32`,
- verifie la dimension attendue `1024`,
- cree l'index FAISS,
- sauvegarde l'index binaire,
- sauvegarde les metadonnees compressees,
- sauvegarde un mapping `doc_id -> position FAISS`.

### Gestion de l'index

Le module `faiss_searcher.py` prend ensuite le relai pour :

- charger l'index et les metadonnees,
- executer une recherche L2 top-k,
- filtrer les candidats par ville, region, tags ou date,
- renvoyer des resultats enrichis avec score de similarite et metadata.

## Validation des index

Les chiffres ci-dessous proviennent des artefacts versionnes et du chargement effectif de l'index FAISS.

### Vue d'ensemble demandee

- 8014 documents disponibles
- 493 villes couvertes
- 1 region (Ile-de-France, graphie normalisee)
- 8486 tags libres uniques (mots-cles libres saisis par les organisateurs dans OpenAgenda)

### Note sur les tags libres

Les 8486 tags sont des mots-cles libres OpenAgenda, non controles par une taxonomie officielle. Chaque organisateur saisit les siens : on y trouve des tags generiques (`culture`, `concert`), des tags techniques (`no-code`, `BTS`), des tags specifiques a un evenement, et quelques donnees parasites (codes courts, caracteres etrangers). Ce n'est pas comparable aux 32 categories officielles d'une taxonomie fermee.

### Correction appliquee sur la region

Le code de collecte normalisait `Île-de-France` en `Ile-de-France` a partir de la version corrigee (27/04/2026). La regeneration du corpus sur cette version produira 1 seule region.

## KPI qualite

### Vue d'ensemble des 41 tests

| Categorie | Tests | Fichier cible | Ce qui est valide |
| --- | --- | --- | --- |
| Tests unitaires | 23 | `tests/unit/` | Fonctions isolees, imports, guardrails, parsing temporel |
| Garde qualite integration | 1 | `test_rag_quality_guard.py` | Corpus RAG complet (8014 docs) vectorisable et coherent |
| Indexation FAISS integration | 17 | `test_faiss_indexing.py` | Index charge, recherche, performance, coherence |
| **Total global** | **41** | | **41/41 PASS (100%)** |

3 warnings de depreciation SWIG observables, non bloquants.

### Pourquoi cette repartition

- **Tests unitaires (23)** : rapides, sans IO, ciblent une fonction a la fois. Permettent de deboguer rapidement un composant isole.
- **Garde qualite (1)** : verifie le JSONL avant indexation. Prerequis logique a toute regeneration du corpus.
- **FAISS integration (17)** : testent l'index reel charge en memoire (31.30 MB). Plus couteux, couvrent la recherche end-to-end et la performance.

### Detail hierarchique des 41 tests

**Tests unitaires — 23 tests repartis sur 4 modules**

| Module | Tests | Sujets couverts |
| --- | --- | --- |
| `test_imports.py` | 3 | Importabilite langchain, SDK Mistral, faiss-cpu |
| `test_rag_chatbot_mistral.py` | 8 | Guardrails, parametres Mistral, format prompt, citation sources, inference tags |
| `test_temporal_deixis.py` | 6 | Expressions deictiques : ce week-end, ce soir, demain soir, apres-demain, aucune |
| `test_vectorize_events_mistral.py` | 6 | Preparation document, rejet URL manquante, vectorisation e2e, cle API |

**Garde qualite dataset — 1 test**

| Test | Ce qui est verifie | Valeur observee |
| --- | --- | --- |
| `test_generated_jsonl_is_vectorizable_and_consistent` | Champs requis, dates, vectorisabilite, URL HTTP, qualite metadata | PASSED sur 8014 docs |

**Classe `TestFAISSIndexCreation` — 3 tests**

| Test | Ce qui est verifie | Valeur observee |
| --- | --- | --- |
| `test_index_exists` | Fichiers d'index presents sur disque | PASSED |
| `test_index_loading` | Chargement en memoire | `ntotal == 8014`, `dim == 1024` |
| `test_index_consistency` | Coherence metadata / index / id_mapping | `8014 == 8014 == 8014` |

**Classe `TestSemanticSearch` — 4 tests**

| Test | Ce qui est verifie | Valeur observee |
| --- | --- | --- |
| `test_basic_search` | Recherche vectorielle top-k | k resultats restitues |
| `test_search_results_ranking` | Ordre des resultats par distance L2 | Distances croissantes confirme |
| `test_search_k_limit` | Nombre de resultats <= k demande | PASSED |
| `test_search_returns_valid_metadata` | Champs requis presents dans chaque resultat | PASSED |

**Classe `TestHybridSearch` — 5 tests**

| Test | Ce qui est verifie | Valeur observee |
| --- | --- | --- |
| `test_hybrid_search_basic` | Recherche hybride sans filtre | PASSED |
| `test_hybrid_search_city_filter` | Tous les resultats matchent la ville demandee | PASSED |
| `test_hybrid_search_tags_filter` | Au moins un tag commun par resultat | PASSED |
| `test_hybrid_search_empty_results` | Filtres tres restrictifs | 0 resultat, sans exception |
| `test_hybrid_search_multiple_filters` | Combinaison ville + tags | PASSED |

**Classe `TestIndexStats` — 3 tests**

| Test | Ce qui est verifie | Valeur observee |
| --- | --- | --- |
| `test_get_stats` | Statistiques d'index coherentes | `unique_tags`, `unique_cities` couverts |
| `test_get_by_id` | Recuperation de document par identifiant | Document retourne |
| `test_nonexistent_id` | ID absent | `None` sans exception |

**Classe `TestPerformance` — 2 tests**

| Test | Ce qui est verifie | Valeur observee (100 requetes, seed fixe) |
| --- | --- | --- |
| `test_search_speed` | Latence par requete < 100 ms | mediane 1.30 ms, p95 2.03 ms, max 3.14 ms |
| `test_index_memory_efficiency` | RAM index sous le seuil fixe | 31.30 MB (ntotal x dim x 4B) |

### KPI performance FAISS (benchmark 100 requetes, seed fixe)

| Strategie | k | Min | Mediane | Moyenne | p95 | Max |
| --- | --- | --- | --- | --- | --- | --- |
| Recherche simple | 5 | 1.09 ms | **1.30 ms** | 1.40 ms | 2.03 ms | 3.14 ms |
| Hybride (k=10, candidats=20) | 20 | 1.12 ms | **1.35 ms** | 1.42 ms | 1.95 ms | 2.80 ms |
| Hybride max (k=50, candidats=100) | 100 | 1.13 ms | **1.29 ms** | 1.34 ms | 1.55 ms | 2.49 ms |

`IndexFlatL2` scanne tous les vecteurs sequentiellement : augmenter k de 5 a 100 ne coute presque rien (~0.1 ms).

### KPI qualite corpus

| Metrique | Valeur | Interpretation |
| --- | --- | --- |
| Documents RAG | 8014 | Corpus regenere sur fenetre corrigee 2025-2027 |
| Vecteurs FAISS | 8014 | 100% alignement RAG -> vecteurs |
| Index FAISS | 31.30 MB | Taille raisonnable, chargement rapide |
| Villes couvertes | 493 | Couverture geographique large IDF |
| Region | 1 (Ile-de-France, normalisee) | Doublons graphiques corriges |
| Tags uniques | 8486 | Tags libres (vocabulaire non controle) |
| Docs sans ville | 5 | Qualite tres bonne (0.06%) |
| Tests PASS | 41/41 | 100% pipeline valide |

## KPI pipeline et dates

### KPI d'alignement pipeline

| Indicateur | Valeur |
| --- | --- |
| Documents RAG | 8014 |
| Lignes du fichier de vecteurs | 8014 |
| IDs uniques dans les vecteurs | 8014 |
| Taux d'alignement RAG -> vecteurs | 100% |
| Metadonnees FAISS | 8014 |
| Entrees de mapping ID | 8014 |

### KPI de fenetre temporelle

| Indicateur | Valeur |
| --- | --- |
| Fenetre forcee de collecte | 25/04/2025 -> 25/04/2027 |
| Evenements commencant avant 25/04/2026 mais conserves | 313 |
| Evenements commencant avant 25/04/2025 mais encore actifs | 142 |
| Evenements se terminant apres 25/04/2027 | 17 |

### Comment lire ces chiffres

La collecte ne retient pas uniquement les evenements dont la date de debut tombe dans la fenetre. Le projet applique une logique de recouvrement d'intervalle : un evenement est conserve si son intervalle `[event_start, event_end]` croise la fenetre cible.

C'est un choix pragmatique et pertinent pour les expositions longues, parcours permanents ou installations s'etendant sur plusieurs mois. Il explique pourquoi certains evenements commencent avant `2025-04-25` ou se terminent apres `2027-04-25` tout en restant presents dans le corpus.

## KPI index FAISS

| Indicateur | Valeur |
| --- | --- |
| Vecteurs indexes | 8014 |
| Dimension embedding | 1024 |
| Type d'index | `IndexFlatL2` |
| Villes uniques | 493 |
| Regions | 1 (Ile-de-France, normalise) |
| Tags libres uniques | 8486 |

## KPI stockage

| Artefact | Taille |
| --- | --- |
| JSONL RAG | 10.26 MB |
| JSONL vecteurs | 192.84 MB |
| Index FAISS | 31.30 MB |
| Metadonnees FAISS | 4.36 MB |
| Mapping ID | 0.34 MB |
| Taille totale artefacts index | 36.00 MB |

## Recherche hybride : principe, metadonnees et choix d'implementation

### Comment la recherche hybride fonctionne ici

La recherche hybride est implemente dans `faiss_searcher.py` avec une logique simple et robuste :

1. calcul des candidats via recherche vectorielle FAISS,
2. filtrage des candidats par metadonnees,
3. retour des meilleurs resultats restants.

Dans l'etat actuel du code, la strategie est :

- recherche vectorielle sur `max(k * 2, 20)` documents,
- filtrage ensuite par :
  - `city`,
  - `region`,
  - `tags`,
  - `after_date`,
  - `before_date`.

### Pourquoi chercher aussi dans les metadonnees

La recherche purement vectorielle est excellente pour la proximite semantique, mais elle ne garantit pas a elle seule :

- le respect d'une ville precise,
- le respect d'une contrainte de date,
- la presence d'un tag metier explicite,
- la remontee de documents avec bonnes metadonnees de contexte.

Le filtre metadata est donc indispensable pour des requetes comme :

- `concert de jazz a Paris`,
- `spectacle a Aubervilliers`,
- `evenement gratuit pour enfant`,
- `festival cet ete en ile de france`.

### Quelles metadonnees sont remontees et exploitees

Les metadonnees chargees depuis l'index contiennent notamment :

- `id`,
- `title`,
- `event_start`,
- `event_end`,
- `city`,
- `region`,
- `country`,
- `tags`,
- `source_record_url`,
- `raw_uid`,
- `text_preview`.

Ces informations servent a deux niveaux :

1. filtrer les documents avant la synthese,
2. enrichir la reponse finale du chatbot avec date, lieu et source.

### Comment les tags sont extraits

Dans `openagenda_culture_france_rag.py`, la fonction `extract_tags` pioche dans sept champs candidats de l'enregistrement OpenAgenda, dans cet ordre de priorite :

```
keywords -> keywords_fr -> tags -> theme -> themes -> themes_fr -> category
```

- Si le champ est une **liste** : chaque element devient un tag.
- Si c'est une **chaine** : elle est decoupee sur `,` `;` `|`.
- Si rien n'est trouve : le tag de secours `culture` est injecte.

Ensuite : deduplication en conservant l'ordre, normalisation `clean_text`.

C'est pour cela qu'il y a **8 486 tags uniques** dans le corpus courant : chaque organisateur saisit librement ses propres mots-cles dans ces champs, sans vocabulaire controle. On y trouve des tags generiques (`culture`, `concert`), des tags techniques (`no-code`, `BTS`), des tags specifiques a un evenement, et quelques donnees parasites (codes courts, caracteres etrangers, phrases longues).

Exemples observes sur le corpus :

```
title: Automatisez votre boutique e-commerce
tags : e-commerce, automatisation, Prestashop, WooCommerce, no-code, low-code, IA...

title: Sous l'ocean
tags : Expositions, energie, environnement, sous-marin

title: Urgence climatique
tags : climat, cgangement climatique (faute), urgence climatique, defi, Humanite
```

### Comment le filtre tags fonctionne dans la recherche hybride

Dans `faiss_searcher.py`, `search_hybrid` fonctionne en deux passes :

```
requete -> embedding -> FAISS top-2k candidats
                              |
                     filtres metadonnees
                    +------------------+
                    | city  (exact)    |
                    | region (exact)   |
                    | tags  (>= 1)     |
                    | after_date       |
                    | before_date      |
                    +------------------+
                              |
                     top-k resultats finaux
```

Pour le filtre tags, la logique est une **intersection d'ensembles** (insensible a la casse) :

```python
doc_tags & filter_tags   # au moins 1 tag commun -> garde
                         # aucun commun          -> elimine
```

**Exemple** : `--tags jazz` garde uniquement les documents dont la liste de tags contient `jazz` (ou `Jazz`, `JAZZ`). Un concert de blues semantiquement proche est elimine si le mot `jazz` n'y figure pas.

**Limite actuelle** : le filtre tag est exact sur la valeur complete. `jazz` ne matche pas `jazz manouche` sauf si l'organisateur a pose `jazz` et `jazz manouche` comme deux tags separes.

## Pourquoi `IndexFlatL2` a ete choisi

### Justification fonctionnelle

Le choix de `IndexFlatL2` est defensable ici pour quatre raisons :

1. le corpus reste de taille moderee avec 8014 vecteurs,
2. la recherche exacte est deja tres rapide,
3. l'absence d'approximation simplifie le debug et la validation,
4. on evite la complexite de parametrage d'indexes approximatifs alors que le gain n'est pas necessaire a cette echelle.

### Mini-benchmark comparatif sur les artefacts courants

Benchmark realise sur 100 requetes echantillonnees a partir des embeddings versionnes :

| Index | Latence moyenne / requete | Recall@10 vs exact | Observation |
| --- | --- | --- | --- |
| `IndexFlatL2` | 0.055 ms | 1.000 | base exacte, reference |
| `IndexHNSWFlat` | 0.018 ms | 0.998 | plus rapide, mais approximation et parametrage supplementaire |
| `IndexIVFFlat` | 0.027 ms | 0.922 | plus rapide que flat, mais perte de rappel plus nette |

### Conclusion technique

Sur un corpus de 8014 vecteurs, `IndexFlatL2` est le meilleur compromis :

- la latence absolue est deja negligeable,
- le rappel est parfait,
- la maintenance reste simple,
- la validation par tests est plus lisible.

`IndexHNSWFlat` deviendrait un candidat credible si le corpus grossissait fortement et que la charge requete augmentait. `IndexIVFFlat` est moins convaincant a cette taille car la perte de rappel est disproportionnee par rapport au gain observe.

## Resultats chatbot : questions, reponses et jugement

Les cas ci-dessous ont ete executes sur le chatbot reel, a partir des artefacts versionnes, sans edition intermediaire des donnees.

### 1. Question : `je cherche une sortie en famille ce week-end en ile de france`

Top documents recuperes sur le corpus regenere :

- `Dimanche au Vert en famille avec enfants de 3 à 10 ans - Parc de l'Ile St Germain à Issy-les-Moulineaux`

Reponse observee :

- le chatbot propose `Dimanche au Vert` a Issy-les-Moulineaux, le 14/09/2025,
- 6 documents sources recuperes, modele `mistral-small-latest`.

Jugement : **partiel / temporel fragile**

La composante `famille` est correcte, la zone IDF aussi, mais l'ancrage `ce week-end` n'est pas fiable dans la CLI actuelle. La question met donc en evidence une limite de chainage entre requete libre et filtres temporels.

### 2. Question : `as-tu un concert de jazz a Paris ?`

Top documents recuperes sur le corpus regenere :

- `Les samedis "Brasil e jazz" / par Jean-Baptiste Loutte` — 23/05/2026 19h30 & 21h30
- `JASS Session / Jam Jazz` — 29/05/2026 20h30
- `Échecs & Jam ! Entrée libre` — 17/05, 03/05, 31/05/2026 de 17h à 22h

Reponse observee :

- plusieurs evenements jazz a Paris sont proposes avec horaires precis,
- les dates, horaires et sources sont correctement restitues,
- 6 documents sources recuperes.

Jugement : **bon**

Le couple intention semantique + filtre implicite de lieu fonctionne bien ici, grace a des metadonnees riches et a des titres tres discriminants.

### 3. Question : `je veux une exposition photo en ile de france`

Top documents recuperes sur le corpus regenere :

- `Expo Photo "c'est la fête !"` — 21/03/2026
- `Expositions archéologiques` — 13/06/2026

Reponse observee :

- le chatbot remonte plusieurs expositions pertinentes,
- il distingue correctement une exposition photo d'une exposition non photo,
- les sources sont bien citees.

Jugement : **bon**

Le systeme tire bien parti des tags et du contenu descriptif.

### 4. Question : `donne moi un evenement gratuit pour enfant`

Top documents recuperes sur le corpus regenere :

- `Les mercredis des possibles` a Saint-Ouen — 27/05/2026

Reponse observee :

- un evenement pour enfant est propose,
- la gratuite n'est pas explicitement demontree par la sortie choisie,
- les champs date / lieu / source sont presents.

Jugement : **acceptable**

L'intention `enfant` est bien couverte, mais le critere `gratuit` gagnerait a etre un filtre metadata explicite plutot qu'une simple inference semantique.

### 5. Question : `je cherche une activite autour des sciences ou de l astronomie`

Top documents recuperes sur le corpus regenere :

- stages `Initiation à l'astronomie` pour 6-8 ans, 9-11 ans et 12-14 ans — 2025/2026
- `Astro junior`

Reponse observee :

- le chatbot propose plusieurs activites tres pertinentes,
- l'astronomie est bien couverte avec des propositions par tranche d'age,
- la reponse s'appuie sur plusieurs sources solides.

Jugement : **tres bon**

Le corpus est bien outille sur ce theme et les tags scientifiques remontent correctement.

### 6. Question : `propose moi un spectacle a Aubervilliers`

Top documents recuperes sur le corpus regenere :

- `La Bamboche à Bambins` au Point Fort d'Aubervilliers — 18/03/2026

Reponse observee :

- la reponse finale propose `La Bamboche à Bambins` au Point Fort d'Aubervilliers avec date et source.

Jugement : **bon**

Le corpus regenere remonte maintenant un spectacle localement pertinent en premier candidat, ce qui est une amelioration par rapport a la version anterieure.

### 7. Question : `y a-t-il un festival de musique cet ete en ile de france ?`

Top documents recuperes sur le corpus regenere :

- `Un temps pour Elles` — 4-5/07/2026
- `Seine en scène` — 21/06/2026

Reponse observee :

- deux festivals d'ete sont proposes avec des dates en juin-juillet 2026,
- les dates sont cohérentes avec l'ete IDF,
- les sources sont bien citees.

Jugement : **bon**

Sur le corpus regenere (8014 docs), la thematique `festival de musique ete` remonte directement des evenements situes en juillet et juin, ce qui est une amelioration nette par rapport a la version anterieure qui proposait des evenements de septembre.

### 8. Question : `donne moi une idee de sortie culturelle a Paris demain soir`

Top documents recuperes sur le corpus regenere :

- `Dessine-moi une œuvre` — 23/05/2026 a Colombes/Paris

Reponse observee :

- une nocturne parisienne est bien proposee,
- la date retournee dans la reponse correspond a `23 mai 2026`,
- cela ne correspond pas a `demain soir` (28/04/2026) — le temporel deictique n'est pas injecte dans le retrieval.

Jugement : **partiel / erreur d'ancrage temporel**

La suggestion culturelle reste credible, mais la recherche n'est pas contrainte temporellement par la deixis `demain soir`.

## Bilan des reponses chatbot

### Forces constatees (corpus 8014 documents)

- bonne performance sur les requetes thematiques : jazz, photo, sciences, astronomie,
- bonnes citations de sources,
- bonne exploitation des tags et des metadonnees de lieu,
- amelioration nette sur les festivals d'ete et les spectacles locaux grace au corpus elargi,
- index FAISS suffisamment riche pour retourner plusieurs candidats exploitables.

### Limites constatees

- le temporel deictique n'est pas relie de facon deterministe a la recherche hybride,
- les criteres `gratuit`, `famille`, `ce week-end`, `demain soir`, `cet ete` gagneraient a devenir des filtres explicites,
- certains cas geographiques montrent un retrieval initial bruite avant que le LLM n'isole une bonne suggestion.

## Filtrage intentions/tags + fallback hors-contexte

L'amelioration implementee repose sur 4 couches dans `MistralRAGChatbot.ask`:

1. **Detection de questions hors-perimetre** avant retrieval.
2. **Detection de questions quantitatives base/corpus** avant retrieval.
3. **Detection de requetes trop generales** avant retrieval.
4. **Inference des intentions metier -> tags** puis filtrage pre-retrieval via `search_hybrid(tags=...)`.

### 1) Fallback hors-contexte

Si la question contient un indice hors-domaine (ex: `meteo`, `temps`, `pluie`) sans indice culturel, le bot repond avec un message fixe:

> Cette question semble hors du perimetre du bot (evenements culturels en Ile-de-France). Je peux t'aider pour des sorties culturelles (concert, exposition, spectacle, activites famille, etc.).

### 2) Fallback quantitatif base de donnees

Si la question demande un comptage/statistiques globales de la base (`combien`, `nombre`, `total`, `kpi` + `base`, `corpus`, `dataset`, etc.), le bot repond avec un fallback fixe:

> Je ne peux pas fournir de statistiques globales fiables sur la base (comptage total, volumes, KPI) via cette interface conversationnelle. Je peux en revanche recommander des evenements concrets.

### 3) Fallback requete trop generale

Si la question demande une liste exhaustive sans filtre utile (type d evenement, ville ou periode), le bot refuse de lancer le retrieval et invite a reformuler.

Patterns detectes :

| Pattern | Exemple |
|---|---|
| `tous les evenements [culturels]` | *"je voudrais tous les evenements culturels a Paris en 2026"* |
| `l ensemble des evenements` | *"donne-moi l ensemble des evenements de la base"* |
| `tout le programme` / `toute la programmation` | *"tout le programme de mai"* |
| `liste complete / exhaustive` | *"une liste complete des concerts"* |

Message retourne :

> Ta demande est trop generale pour que je puisse selectionner des evenements pertinents. Je donne de bien meilleurs resultats sur des requetes precises : un type d evenement, une ville ou une periode. Exemples : 'concert de jazz a Paris ce week-end', 'exposition photo en mai a Versailles', 'spectacle famille a Montreuil'.

**Justification** : une requete sans filtre recupererait les k premiers voisins vectoriels les plus proches du vecteur d une phrase tres generique, produisant un bruit incoherent plutot qu une vraie recommandation.

### 4) Filtrage pre-retrieval base sur les tags

Le pipeline applique cette strategie:

1. Normaliser la question (minuscule + suppression accents).
2. Extraire des tokens candidats depuis la question.
3. Mapper les intentions metier vers des tags (`INTENT_TO_TAGS`) :
   - `jazz` -> `jazz`, `musique`, `concert`
   - `famille` -> `famille`, `enfant`
   - `exposition` -> `exposition`, `photo`, `art`
   - `science` -> `science`, `astronomie`
4. Ne conserver que les tags reellement presents dans l'index FAISS (`available_tags`).
5. Appeler `search_hybrid(..., tags=effective_tags)`.
6. Si aucun document ne sort (sur-filtrage), fallback automatique vers `search_hybrid(..., tags=None)`.

Ce mecanisme utilise donc bien **tout le catalogue de tags disponible dans les metadonnees** (pas une liste figee uniquement).

## Interface Streamlit — lancement, fonctionnement et panneau de fiabilite

### Lancement

Depuis le dossier `Pull-Events/` avec l'environnement conda actif :

```bash
C:/Users/karap/anaconda3/envs/LLMRag/python.exe -m streamlit run PullEventsIDFBot.py
```

L'interface s'ouvre dans le navigateur a l'adresse `http://localhost:8501`.

### Fonctionnement general

`PullEventsIDFBot.py` est une interface conversationnelle construite avec [Streamlit](https://streamlit.io). Elle encapsule l'integralite du pipeline RAG et expose un formulaire de recherche libre en langage naturel.

Flux complet d'une requete :

```text
Question utilisateur
       |
       v
[Guardrails] hors-perimetre ? quantitatif ? trop general ?
       |                               |
   oui -> reponse fixe             non -> pipeline RAG
                                        |
                              [Temporal deixis]
                          detection fenetre UTC
                          (ce soir, demain, en mai...)
                                        |
                              [Inference tags]
                          question -> tokens -> INTENT_TO_TAGS
                          filtres : jazz, famille, expo...
                                        |
                              [Embedding Mistral]
                          vectorisation de la question
                                        |
                              [FAISS top-K*5]
                          candidats les plus proches
                                        |
                              [Filtres metadata]
                          ville, region, tags, dates
                                        |
                              [Reranking]
                          +bonus tags +bonus ville
                                        |
                              [Top-K documents]
                          injectes dans le prompt LLM
                                        |
                              [Generation Mistral]
                          reponse contrainte par SYSTEM_PROMPT
                                        |
                              [Affichage + indicateurs]
```

### Zone de reglages (sidebar gauche)

| Reglage | Role | Recommandation |
|---|---|---|
| **Temperature** (0.0-1.0) | Creativite du LLM lors de la generation. 0 = tres factuel, 1 = tres creatif. N'affecte pas le retrieval. | 0.1 - 0.3 |
| **Nombre de documents consultes** (k) | Combien de documents sont injectes dans le contexte envoye au LLM. | 5 - 8 |

Le slider k commande directement le nombre de documents que le LLM voit. Un k trop bas risque de manquer des evenements pertinents ; un k trop eleve noie le LLM dans un contexte trop grand.

### Zone de resultat (partie centrale)

Apres chaque recherche, les zones suivantes apparaissent :

1. **Reponse** : texte genere par le LLM, dates converties en format francais.
2. **Note utilisateur** (si necessaire) : conseil de reformulation si le retrieval est partiel.
3. **Panneau de fiabilite** : 5 metriques + fenetre temporelle + tags actifs.
4. **Documents utilises** (expander) : detail de chaque document source avec titre, ville, periode, score et URL.
5. **Prompt envoye au LLM** (expander discret) : prompt complet reel, incluant le prompt systeme et le contexte avec les vrais documents recuperes — utile pour comprendre exactement ce que le modele a vu.
6. **Historique** (expander) : toutes les recherches precedentes de la session.

### Guardrails — requetes refusees avant retrieval

Trois types de questions sont interceptes et refus avant toute requete FAISS :

| Type | Detection | Exemple |
|---|---|---|
| **Hors domaine** | Indices hors-culture sans mot culturel | *"quelle est la meteo a Paris ?"* |
| **Statistiques base** | `combien` + `evenements/base/corpus` | *"combien d'evenements contient la base ?"* |
| **Requete trop large** | Patterns `tous les evenements`, `liste complete`... | *"tous les evenements culturels en 2026"* |

Ces requetes ne coutent aucun appel API et retournent immediatement un message d'orientation.

### Prompt envoye au LLM

Le prompt se compose de deux parties :

**Prompt systeme (fixe)** :

```
Tu es un assistant RAG specialise dans les recommandations d evenements culturels en Ile-de-France.
REGLES STRICTES :
  1) Utilise UNIQUEMENT les informations presentes dans les documents du contexte fourni.
  2) N ajoute AUCUN detail, date, lieu ou titre qui ne figure pas explicitement dans le contexte.
  3) Si une information n est pas disponible dans le contexte, ecris "Information non disponible dans les documents."
  4) Ne formule AUCUNE hypothese sur ce qui pourrait exister hors du contexte.
  5) Cite explicitement les sources (URL) en fin de reponse pour chaque evenement mentionne.
```

**Message utilisateur (dynamique)** :

```
Question: <question de l utilisateur>

Contexte:
[1] Titre evenement (Ville) - 2026-05-10
Description / resume de l evenement...
Source: https://openagenda.com/...

[2] ...
[k] ...
```

Le prompt reel de chaque requete est visible dans l'interface via l'expander **"🔬 Prompt envoye au LLM"** (affiche sous les sources, ferme par defaut).

### Panneau de fiabilite

| Indicateur | Signification | Seuils |
|---|---|---|
| **Niveau** | Estimation globale | 🟢 score>=0.72 + 2 sources + 70% dates ; 🟠 score>=0.60 ; 🔴 en-dessous ; ⚪ hors-perimetre |
| **Docs utilises** | Documents injectes dans le prompt | = valeur du slider k |
| **Sources distinctes** | Nombre d URL OpenAgenda differentes | Plus eleve = reponse mieux recoupee |
| **Score retrieval moyen** | Proximite semantique question/docs | 0 a 1 |
| **Couverture dates** | Part des docs avec dates exploitables | 100% = filtrage temporel efficace |
| **Fenetre temporelle** | Expression temporelle detectee | Ex. `ce_weekend`, `demain_soir`, `en_mai` |
| **Tags actifs** | Tags inferences depuis la question | Filtres appliques en pre-retrieval |

### Role des tags dans la recherche hybride

Les tags inferres participent a deux niveaux :

1. **Filtre dur** : un document est elimine si aucun de ses tags ne correspond a ceux de la requete.
2. **Bonus reranking** : `+0.15` si >= 2 tags matchent, `+0.08` si 1 seul, `+0.10` si ville exacte.

Si les tags inferres sont trop restrictifs (0 resultat), un fallback automatique relance sans filtre de tags.

### Inferences temporelles supportees

| Categorie | Exemples reconnus |
|---|---|
| Deictiques relatifs | *ce soir, demain, apres-demain, ce week-end, cette semaine, semaine prochaine* |
| Soirs | *demain soir, apres-demain soir, ce soir* |
| Mois explicites | *en mai, en mai 2026, pour juin, au mois de mars 2027* |
| Saisons | *en ete, cet ete, en ete 2026, au printemps, en automne, en hiver 2026* |

### Expanders de bas de page (aide contextuelle)

- **📚 Comment le bot fonctionne** : flux complet avec les 3 guardrails et les etapes du pipeline.
- **🔧 Construction du prompt envoye au LLM** : prompt systeme verbatim, format du contexte, role de k, de la temperature et des tags.
- **📏 Comment lire les indicateurs de fiabilite** : guide detaille de chaque metrique.

### Limites de l'interface

- Pas de memoire inter-sessions (l'historique est perdu a la fermeture).
- Pas de filtre geographique manuel (la ville est inferee depuis la question).
- La recherche sans expression temporelle n'est pas contrainte dans le temps : les evenements passes peuvent remonter.

## Evaluation RAGAS (hallucinations et fiabilite)

Cette section mesure objectivement la qualite generation/retrieval **apres** ajout du filtrage intentions/tags et des fallbacks.

### Metriques utilisees

- `faithfulness`
- `context_utilization`
- `context_precision`
- `context_recall`
- ~~`answer_relevancy`~~ *(desactivee — a implementer ulterieurement)*

### Protocole applique

1. Jeu de 7 questions metier (question meteo retiree, question quantitative retiree car traitee par fallback).
2. Passage de chaque question dans le pipeline reel (`MistralRAGChatbot.ask`).
3. Capture de `answer` et `contexts` reellement recuperes.
4. Construction d'une `ground_truth` silver (reference automatique basee uniquement sur les contextes).
5. Evaluation RAGAS sur `question/answer/contexts/ground_truth`.

Script: `tools/diagnostic/ragas_eval_pull_events.py`

Commande de reproduction (depuis `Pull-Events/`) :

```bash
C:/Users/karap/anaconda3/envs/LLMRag/python.exe tools/diagnostic/ragas_eval_pull_events.py --env-file "C:/Users/karap/OpenClassRooms/projet11/coursEtExos/8532116-mettez-en-place-un-rag-pour-un-llm/.env" --k 6
```

### Baseline (avant améliorations — 1 run)

| Metrique | Score |
| --- | ---: |
| `faithfulness` | 0.5932 |
| `context_utilization` | 0.9062 |
| `context_precision` | 0.6238 |
| `context_recall` | 0.8524 |

### Resultats post-améliorations (3 runs — moyenne ± écart-type, rerun du 28/04/2026)

Commande de reproduction :

```bash
Set-Location Pull-Events
C:/Users/karap/anaconda3/envs/LLMRag/python.exe tools/diagnostic/ragas_eval_pull_events.py \
  --env-file "C:/Users/karap/OpenClassRooms/projet11/coursEtExos/8532116-mettez-en-place-un-rag-pour-un-llm/.env" \
  --k 6 --runs 3
```

| Metrique | Moy. | ±std | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| `faithfulness` | 0.9122 | 0.0621 | 0.8571 | 0.9796 |
| `context_utilization` | 0.7165 | 0.0187 | 0.6950 | 0.7284 |
| `context_precision` | 0.7130 | 0.0484 | 0.6776 | 0.7681 |
| `context_recall` | 0.7698 | 0.0576 | 0.7211 | 0.8333 |
| ~~`answer_relevancy`~~ | *(a implementer)* | | | |

### Tableau des gains (baseline → post-améliorations)

| Metrique | Baseline | Post-amélio | Delta | % |
| --- | ---: | ---: | ---: | ---: |
| `faithfulness` | 0.5932 | **0.9122** | **+0.3190** | **+53.8% ✅** |
| `context_utilization` | 0.9062 | 0.7165 | -0.1897 | -20.9% ⚠️ |
| `context_precision` | 0.6238 | 0.7130 | +0.0892 | +14.3% ✅ |
| `context_recall` | 0.8524 | 0.7698 | -0.0826 | -9.7% ⚠️ |

### Analyse des résultats

**Gain principal** : `faithfulness` +53.8% — le durcissement du SYSTEM_PROMPT anti-hallucination, le reranking et le branchement des filtres ont sensiblement réduit les affirmations non ancrées dans le contexte.

Les 3 runs ci-dessus correspondent au pipeline courant, après correctif du filtre temporel par chevauchement d'intervalle.

Correctif applique ensuite dans `search_hybrid` :

- conserver un evenement si son intervalle `[event_start, event_end]` croise la fenetre `[after_date, before_date]`;
- equivalent logique : `event_end >= after_date` et `event_start <= before_date`.

Ce correctif evite d'exclure a tort les evenements recurrents (date de debut ancienne, date de fin future), et corrige le cas `sortie en famille ce week-end` qui retournait 0 document.

**Note** : la variabilité inter-runs reste moderee (std ≤ 0.07), avec une stabilite particulierement bonne sur `context_utilization`.

### Limitation technique — answer_relevancy desactivee en version prod

- `answer_relevancy` est conservee en commentaire dans le script et sera reactivee ulterieurement.
- Le choix prod actuel privilegie des metriques stables et reproductibles avec l'environnement present.

## Améliorations RAG implémentées (phase 2)

Ces améliorations ont été appliquées après la baseline RAGAS initiale :

### 1) Enrichissement INTENT_TO_TAGS

Le dictionnaire d'intentions métier a été élargi pour couvrir davantage de synonymes :

| Ajout | Tags associés |
|---|---|
| `ado` / `adolescent` / `jeune` | `jeune public`, `adolescent`, `famille` |
| `musee` | `musee`, `exposition`, `art` |
| `cirque` | `cirque`, `spectacle` |
| `humour` | `humour`, `spectacle` |
| `sortie` | `sortie`, `famille`, `loisir` |
| `balade` / `promenade` | `balade`, `nature`, `plein air` |
| `litterature` / `lecture` | `litterature`, `lecture` |
| `nature` / `plein air` | `nature`, `plein air`, `balade` |
| `patrimoine` | `patrimoine` |
| `gastronomie` | `gastronomie` |
| `sport` | `sport` |
| `libre` / `entree libre` | `gratuit` |

### 2) Reranking léger dans search_hybrid

Avant : `top_k_search = max(k*2, 20)` candidats, retour direct des premiers k après filtres.

Après : `top_k_search = max(k*5, 50)` candidats, puis reranking par score combiné :

- `score = similarity_score`
- `+0.15` si ≥ 2 tags de l'intention matchent dans le document
- `+0.08` si 1 seul tag matche
- `+0.10` si la ville exacte est présente

Les résultats sont triés par `rerank_score` décroissant avant de retourner top-k.

### 3) Durcissement SYSTEM_PROMPT anti-hallucination

Le prompt système a été renforcé avec 5 règles explicites :

1. Utiliser **uniquement** les documents du contexte fourni.
2. N'ajouter aucun détail non présent littéralement dans le contexte.
3. Si information absente → écrire *"Information non disponible dans les documents."*
4. Ne formuler aucune hypothèse hors contexte.
5. Citer les sources (URL) systématiquement.

Impact mesuré : `faithfulness` +53.8% (0.59 → 0.91).

### 4) Branchement automatique du filtre temporel

`temporal_deixis.infer_temporal_window()` est maintenant appelée automatiquement dans `MistralRAGChatbot.ask()` si aucun `after_date`/`before_date` n'est fourni explicitement. La fenêtre UTC déduite est passée à `search_hybrid`.

Expressions reconnues :

| Catégorie | Exemples |
|---|---|
| Déictiques relatifs | *"ce week-end"*, *"demain soir"*, *"aujourd'hui"*, *"cette semaine"*, *"semaine prochaine"*, *"après-demain"*, *"ce soir"* |
| Mois explicites (avec ou sans année) | *"en mai"*, *"en mai 2026"*, *"de mai 2026"*, *"pour juin"*, *"au mois de mars 2027"* |
| Saisons (avec ou sans année) | *"en été"*, *"cet été"*, *"en été 2026"*, *"au printemps"*, *"en automne"*, *"en hiver 2026"* |

Pour les mois explicites, la fenêtre est calculée sur le mois entier (du 1er au dernier jour). Si l'année est absente, l'année courante est utilisée ; si le mois est déjà passé, l'année suivante est automatiquement inférée.

Le filtrage date est applique via recouvrement d'intervalle (`event_start` / `event_end`) pour ne pas exclure les evenements encore en cours.

Un fallback automatique sans filtre de tags est en place (`ask()` retry avec `tags=None`) mais pas sans filtre temporel — si le corpus ne contient aucun événement dans la fenêtre détectée, 0 résultats seront retournés.

### 5) Mise à jour chatbot_cli.py

- `--verbose` : affiche la fenêtre temporelle auto-détectée, le nombre de docs récupérés et leur `rerank_score`.
- Sortie `--json` enrichie avec `title`, `city`, `rerank_score`, `tags`, `url` par document.
- `--tags` documenté comme override de l'auto-inférence.
- Encodage console Windows : affichage ASCII (`->`) dans les logs verbose pour eviter les erreurs `cp1252`.

### 6) Affichage Streamlit en format français

- Dans `PullEventsIDFBot.py`, les dates affichées dans les résultats sont normalisées au format français `JJ/MM/AAAA`.
- Les périodes d'événements sont présentées en clair (`du JJ/MM/AAAA au JJ/MM/AAAA`) et passent en `JJ/MM/AAAA HH:MM` quand les heures sont disponibles.
- Les dates présentes dans le texte de réponse affiché sont également converties en format français.

---

## Rapport technique complet

### Ce qui est techniquement solide dans la solution

1. la collecte est defensive et filtre deja fortement le bruit,
2. le corpus final est compact et bien aligne avec la vectorisation,
3. l'index exact FAISS est simple a valider et tres rapide,
4. la recherche hybride repose sur des metadonnees directement exploitables,
5. la suite de tests couvre la qualite du JSONL, le chargement de l'index, la recherche et une partie de la performance.

### Ce qui merite d'etre surveille

1. les variantes d'ecriture de region restent visibles dans les stats,
2. l'efficacite du filtre temporel depend de la qualite des champs `event_start`/`event_end`,
3. la CLI chatbot expose bien les fenetres deictiques (`--verbose`) mais pas encore un mode de fallback temporel automatique,
4. certains attributs metier comme la gratuite ne sont pas modelises comme filtres metadata natifs.

## Pistes d'amélioration futures

1. ~~**Mois explicites dans le filtre temporel**~~ ✅ *Implémenté le 27/04/2026 — `temporal_deixis.py` gère désormais "en mai 2026", "pour juin", "au mois de mars", etc.*
2. ~~**Détection des requêtes trop générales**~~ ✅ *Implémenté le 28/04/2026 — `_is_too_broad_question` détecte et refuse les requêtes sans filtre utile.*
3. **Fallback temporel** : si le filtre temporel auto retourne 0 docs, relancer sans filtre de dates (corpus historique).
4. **Re-run RAGAS post-correctif date** : mesurer proprement le gain apres correction du chevauchement d'intervalle.
5. **`answer_relevancy`** : réactiver la métrique RAGAS quand la version >= 0.2.x sera compatible avec l'environnement.
6. **Gratuit comme filtre metadata natif** : modéliser la gratuité comme attribut booléen dans les métadonnées FAISS.
7. **Corpus vivant** : pipeline de mise à jour incrémentale pour que le corpus reste à jour avec les événements futurs.



### Pourquoi ne pas l'implementer dans ce lot de travaux

Le besoin explicite de ce lot de travaux est de corriger la fenetre de collecte et de documenter la solution sans ouvrir un chantier de modification fonctionnelle plus large. Le chainage temporel chatbot est une amelioration adjacente, utile, mais distincte de la correction de la collecte.

## Conclusion

La solution est maintenant documentee de bout en bout.

Les points a retenir sont les suivants :

- la fenetre de collecte source a ete corrigee vers `25/04/2025 -> 25/04/2027`,
- le corpus a ete regenere sur cette fenetre : 8014 documents RAG, 8014 vecteurs, index FAISS 31.30 MB,
- la validation technique ne montre aucune regression sur le pipeline existant (41/41 tests PASS),
- les artefacts regeneres sont coherents a 100%,
- la recherche hybride est solide sur les criteres thematiques et geographiques,
- les reponses chatbot sur festivals d'ete et spectacles locaux sont meilleures sur le corpus elargi,
- le chainage temporel deictique est actif, avec filtre par chevauchement d'intervalle pour la recherche datee.

## Annexes utiles

### Commandes resumees

```bash
conda activate LLMRag
pip install -r requirements.txt
C:/Users/karap/anaconda3/envs/LLMRag/python.exe openagenda_culture_france_rag.py
C:/Users/karap/anaconda3/envs/LLMRag/python.exe vectorize_events_mistral.py --env-file "C:/Users/karap/OpenClassRooms/projet11/coursEtExos/8532116-mettez-en-place-un-rag-pour-un-llm/.env"
C:/Users/karap/anaconda3/envs/LLMRag/python.exe index_events_faiss.py
C:/Users/karap/anaconda3/envs/LLMRag/python.exe -m pytest tests/unit -q
C:/Users/karap/anaconda3/envs/LLMRag/python.exe -m pytest tests/integration/test_rag_quality_guard.py -q
C:/Users/karap/anaconda3/envs/LLMRag/python.exe -m pytest tests/integration/test_faiss_indexing.py -q
C:/Users/karap/anaconda3/envs/LLMRag/python.exe chatbot_cli.py --question "as-tu un concert de jazz a Paris ?"
```

### Artefacts de reference

- `data/evenements_publics_openagenda_culture_ile_de_france_rag.jsonl`
- `data/evenements_publics_openagenda_culture_ile_de_france_vectors.jsonl`
- `data/faiss_index.idx`
- `data/faiss_metadata.pkl`
- `data/faiss_id_mapping.pkl`