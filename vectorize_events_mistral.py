from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from mistralai.client import MistralClient


DEFAULT_INPUT_PATH = "data/evenements_publics_openagenda_culture_ile_de_france_rag.jsonl"
DEFAULT_OUTPUT_PATH = "data/evenements_publics_openagenda_culture_ile_de_france_vectors.jsonl"
DEFAULT_MODEL = "mistral-embed"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MIN_TEXT_CHARS = 120


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def first_non_empty(record: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for key in keys:
        value = record.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def to_tag_list(value: Any) -> List[str]:
    if isinstance(value, list):
        raw = [clean_text(v) for v in value if clean_text(v)]
    elif value:
        raw = [part.strip() for part in re.split(r"[;,|]", clean_text(value)) if part.strip()]
    else:
        raw = []

    tags: List[str] = []
    seen = set()
    for tag in raw:
        lowered = tag.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        tags.append(tag)
    return tags


def build_fallback_text(record: Dict[str, Any]) -> str:
    title = clean_text(first_non_empty(record, ["title", "title_fr", "name"]))
    summary = clean_text(
        first_non_empty(
            record,
            [
                "summary",
                "description",
                "description_fr",
                "longdescription",
                "longdescription_fr",
                "lead_text",
            ],
        )
    )
    start = clean_text(first_non_empty(record, ["event_start", "firstdate_begin"]))
    end = clean_text(first_non_empty(record, ["event_end", "lastdate_end"]))
    location = clean_text(first_non_empty(record, ["location_name", "location"]))
    city = clean_text(first_non_empty(record, ["city", "location_city"]))
    region = clean_text(first_non_empty(record, ["region", "location_region"]))
    tags = to_tag_list(
        first_non_empty(record, ["tags", "keywords", "keywords_fr", "themes", "themes_fr", "theme", "category"])
    )

    parts = [
        f"Titre: {title}" if title else "",
        f"Resume: {summary}" if summary else "",
        f"Debut: {start}" if start else "",
        f"Fin: {end}" if end else "",
        f"Lieu: {location}" if location else "",
        f"Ville: {city}" if city else "",
        f"Region: {region}" if region else "",
        "Pays: France",
        f"Tags: {', '.join(tags)}" if tags else "",
    ]
    return "\n".join(part for part in parts if part)


def normalize_source_url(record: Dict[str, Any]) -> Optional[str]:
    url = clean_text(first_non_empty(record, ["source_record_url", "canonicalurl", "url", "shareurl", "html"]))
    if url.startswith(("http://", "https://")):
        return url
    return None


def build_doc_id(record: Dict[str, Any], fallback_index: int) -> str:
    if record.get("id"):
        return clean_text(record["id"])

    raw_uid = clean_text(first_non_empty(record, ["uid", "slug", "raw_uid"]))
    if raw_uid:
        return f"evenements-publics-openagenda:{raw_uid}"
    return f"evenements-publics-openagenda:generated-{fallback_index}"


def prepare_embedding_document(
    record: Dict[str, Any],
    fallback_index: int,
    min_text_chars: int = DEFAULT_MIN_TEXT_CHARS,
) -> Optional[Dict[str, Any]]:
    doc_id = build_doc_id(record, fallback_index)

    content = clean_text(record.get("content"))
    text = content if len(content) >= min_text_chars else build_fallback_text(record)
    text = clean_text(text)
    if len(text) < min_text_chars:
        return None

    title = clean_text(first_non_empty(record, ["title", "title_fr", "name"])) or "Evenement culturel"
    source_url = normalize_source_url(record)
    if not source_url:
        return None

    tags = to_tag_list(first_non_empty(record, ["tags", "keywords", "keywords_fr", "themes", "themes_fr", "theme", "category"]))
    metadata = {
        "title": title,
        "event_start": clean_text(first_non_empty(record, ["event_start", "firstdate_begin"])) or None,
        "event_end": clean_text(first_non_empty(record, ["event_end", "lastdate_end"])) or None,
        "city": clean_text(first_non_empty(record, ["city", "location_city"])) or None,
        "region": clean_text(first_non_empty(record, ["region", "location_region"])) or None,
        "country": clean_text(first_non_empty(record, ["country", "country_fr"])) or "France",
        "tags": tags,
        "source_dataset": clean_text(first_non_empty(record, ["source_dataset", "dataset"])) or "evenements-publics-openagenda",
        "source_record_url": source_url,
        "raw_uid": clean_text(first_non_empty(record, ["uid", "raw_uid", "slug"])) or None,
    }

    return {
        "id": doc_id,
        "text": text,
        "metadata": metadata,
    }


def chunked(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]


def embed_texts(client: Any, texts: List[str], model: str, max_retries: int = 3) -> List[List[float]]:
    """Appel API avec retry exponentiel en cas d'erreur."""
    last_error: Exception = RuntimeError("embed_texts: aucune tentative effectuee")
    for attempt in range(1, max_retries + 1):
        try:
            response = client.embeddings(model=model, input=texts)
            return [entry.embedding for entry in response.data]
        except Exception as exc:
            last_error = exc
            wait = 2 ** attempt
            print(f"  [RETRY {attempt}/{max_retries}] Erreur API: {exc} — attente {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Echec apres {max_retries} tentatives: {last_error}") from last_error


def _load_already_written_ids(output_file: Path) -> Set[str]:
    """Lit le fichier de sortie existant et retourne les IDs deja ecrits."""
    ids: Set[str] = set()
    if not output_file.exists():
        return ids
    with output_file.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                doc_id = rec.get("id")
                if doc_id:
                    ids.add(doc_id)
            except json.JSONDecodeError:
                pass
    return ids


def _compact_output_file(output_file: Path, allowed_ids: Set[str]) -> Set[str]:
    """Ne conserve dans le fichier de sortie que les IDs du corpus courant, une seule fois."""
    if not output_file.exists():
        return set()

    kept_lines: List[str] = []
    kept_ids: Set[str] = set()
    original_non_empty = 0

    with output_file.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            original_non_empty += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_id = rec.get("id")
            if not doc_id or doc_id in kept_ids or doc_id not in allowed_ids:
                continue

            kept_ids.add(doc_id)
            kept_lines.append(json.dumps(rec, ensure_ascii=False))

    if len(kept_lines) != original_non_empty:
        with output_file.open("w", encoding="utf-8") as fh:
            for line in kept_lines:
                fh.write(line + "\n")
        print(
            f"[CLEANUP] Sortie vectorisee compactee: {original_non_empty} -> {len(kept_lines)} lignes utiles.",
            flush=True,
        )

    return kept_ids


def _count_lines(path: Path) -> int:
    """Compte les lignes non vides d'un fichier JSONL."""
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def load_env_value(env_file: Path, key: str) -> Optional[str]:
    if not env_file.exists():
        return None

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() != key:
            continue
        parsed = value.strip().strip('"').strip("'")
        return parsed or None
    return None


def resolve_api_key(explicit_api_key: Optional[str], env_file: Optional[str]) -> str:
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()

    env_value = os.environ.get("MISTRAL_API_KEY", "").strip()
    if env_value:
        return env_value

    if env_file:
        from_file = load_env_value(Path(env_file), "MISTRAL_API_KEY")
        if from_file:
            return from_file

    raise EnvironmentError(
        "MISTRAL_API_KEY manquante. Definir la variable d'environnement, "
        "ou passer --api-key, ou --env-file avec une entree MISTRAL_API_KEY."
    )


def vectorize_jsonl(
    input_path: str,
    output_path: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_docs: Optional[int] = None,
    min_text_chars: int = DEFAULT_MIN_TEXT_CHARS,
    client: Any = None,
    progress_every: int = 5,
) -> Dict[str, int]:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Fichier source introuvable: {input_path}")
    if batch_size <= 0:
        raise ValueError("batch_size doit etre strictement positif")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    mistral_client = client or MistralClient(api_key=api_key)

    stats: Dict[str, int] = {
        "read": 0,
        "prepared": 0,
        "skipped": 0,
        "resumed": 0,
        "embedded": 0,
        "written": 0,
        "errors": 0,
    }

    # --- Phase 1 : chargement et preparation ---
    print("[PHASE 1] Lecture et preparation des documents...", flush=True)
    t_load_start = time.time()
    prepared_docs: List[Dict[str, Any]] = []

    with input_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_docs is not None and stats["read"] >= max_docs:
                break

            line = line.strip()
            if not line:
                continue

            stats["read"] += 1
            record = json.loads(line)
            prepared = prepare_embedding_document(
                record=record,
                fallback_index=stats["read"],
                min_text_chars=min_text_chars,
            )
            if not prepared:
                stats["skipped"] += 1
                continue

            prepared_docs.append(prepared)
            stats["prepared"] += 1

    valid_ids = {doc["id"] for doc in prepared_docs}
    already_written = _compact_output_file(output_file, valid_ids)
    if already_written:
        print(f"[REPRISE] {len(already_written)} documents deja vectorises — ils seront ignores.", flush=True)
    stats["resumed"] = len(already_written)
    prepared_docs = [doc for doc in prepared_docs if doc["id"] not in already_written]
    stats["prepared"] = len(prepared_docs)

    t_load = time.time() - t_load_start
    total_to_embed = len(prepared_docs)
    print(
        f"[PHASE 1] Termine en {_fmt_duration(t_load)} — "
        f"lus={stats['read']}, a vectoriser={total_to_embed}, "
        f"ignores={stats['skipped']}, deja ecrits={len(already_written)}",
        flush=True,
    )

    if not prepared_docs:
        if already_written:
            print("[INFO] Tous les documents sont deja vectorises. Rien a faire.", flush=True)
            return stats
        raise ValueError("Aucun document exploitable pour l'embedding.")

    # Estimation du temps total (base sur ~0.5s par batch en moyenne)
    n_batches = (total_to_embed + batch_size - 1) // batch_size
    est_seconds = n_batches * 0.6
    print(
        f"[PHASE 2] Debut vectorisation — {total_to_embed} docs, "
        f"{n_batches} batchs de {batch_size}, "
        f"duree estimee ~{_fmt_duration(est_seconds)}",
        flush=True,
    )

    # --- Phase 2 : vectorisation par batchs ---
    t_embed_start = time.time()
    batch_num = 0

    try:
        with output_file.open("a", encoding="utf-8") as out:
            for batch in chunked(prepared_docs, batch_size):
                batch_num += 1
                texts = [doc["text"] for doc in batch]

                try:
                    embeddings = embed_texts(mistral_client, texts, model=model)
                except RuntimeError as exc:
                    stats["errors"] += 1
                    print(f"  [ERREUR batch {batch_num}/{n_batches}] {exc} — batch ignore.", flush=True)
                    continue

                if len(embeddings) != len(batch):
                    stats["errors"] += 1
                    print(
                        f"  [ERREUR batch {batch_num}/{n_batches}] "
                        f"Taille embeddings ({len(embeddings)}) != taille batch ({len(batch)}) — ignore.",
                        flush=True,
                    )
                    continue

                stats["embedded"] += len(embeddings)

                for doc, vector in zip(batch, embeddings):
                    out.write(
                        json.dumps(
                            {"id": doc["id"], "text": doc["text"], "embedding": vector, "metadata": doc["metadata"]},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    stats["written"] += 1
                out.flush()

                # Affichage progression
                if batch_num % progress_every == 0 or batch_num == n_batches:
                    elapsed = time.time() - t_embed_start
                    docs_done = stats["written"]
                    pct = docs_done / total_to_embed * 100 if total_to_embed else 0
                    speed = docs_done / elapsed if elapsed > 0 else 0
                    remaining = (total_to_embed - docs_done) / speed if speed > 0 else 0
                    print(
                        f"  [{batch_num:4d}/{n_batches}] "
                        f"{docs_done}/{total_to_embed} docs ({pct:5.1f}%) | "
                        f"ecrits={stats['written']} erreurs={stats['errors']} | "
                        f"ecoule={_fmt_duration(elapsed)} ETA={_fmt_duration(remaining)}",
                        flush=True,
                    )

    except KeyboardInterrupt:
        elapsed = time.time() - t_embed_start
        print(
            f"\n[INTERRUPTION] Arret demande apres {_fmt_duration(elapsed)}. "
            f"Ecrits={stats['written']}/{total_to_embed}. "
            f"Relancez la meme commande pour reprendre.",
            flush=True,
        )
        return stats

    elapsed_total = time.time() - t_embed_start
    print(
        f"[PHASE 2] Termine en {_fmt_duration(elapsed_total)} — "
        f"embeddes={stats['embedded']}, ecrits={stats['written']}, erreurs={stats['errors']}",
        flush=True,
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vectorise les evenements JSONL avec les embeddings Mistral.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Chemin du fichier JSONL source.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Chemin du fichier JSONL vectorise.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modele d'embedding Mistral.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Taille des lots d'embedding.")
    parser.add_argument("--max-docs", type=int, default=None, help="Limite optionnelle sur le nombre de documents lus.")
    parser.add_argument("--min-text-chars", type=int, default=DEFAULT_MIN_TEXT_CHARS, help="Longueur minimale d'un texte vectorisable.")
    parser.add_argument("--api-key", default=None, help="Cle API Mistral (optionnel, sinon variable d'environnement).")
    parser.add_argument("--env-file", default=None, help="Chemin d'un fichier .env contenant MISTRAL_API_KEY.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = resolve_api_key(explicit_api_key=args.api_key, env_file=args.env_file)

    print(f"[START] input={args.input} | output={args.output} | model={args.model} | batch={args.batch_size}", flush=True)
    t_start = time.time()

    stats = vectorize_jsonl(
        input_path=args.input,
        output_path=args.output,
        api_key=api_key,
        model=args.model,
        batch_size=args.batch_size,
        max_docs=args.max_docs,
        min_text_chars=args.min_text_chars,
    )

    total_elapsed = time.time() - t_start
    total_written = stats['written'] + stats.get('resumed', 0)
    print(
        f"\n[DONE] lus={stats['read']} | prepares={stats['prepared']} | "
        f"ignores={stats['skipped']} | embeddes={stats['embedded']} | "
        f"ecrits={stats['written']} | erreurs={stats['errors']} | "
        f"duree totale={_fmt_duration(total_elapsed)}"
    )
    print(f"Fichier de sortie: {args.output} ({total_written} docs au total)")


if __name__ == "__main__":
    main()
