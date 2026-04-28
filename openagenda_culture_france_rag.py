from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from dateutil import parser as dateparser


BASE_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records/"
API_KEY = os.environ.get("API_KEY")

WINDOW_START_UTC = datetime(2025, 4, 25, 0, 0, 0, tzinfo=timezone.utc)
WINDOW_END_UTC = datetime(2027, 4, 25, 23, 59, 59, tzinfo=timezone.utc)
OPENAGENDA_MAX_OFFSET = 10_000

HEADERS = {
    "Content-Type": "application/json",
}
if API_KEY:
    HEADERS["Authorization"] = f"Apikey {API_KEY}"

# Mots-clés métier pour garder les événements "culture".
CULTURE_TERMS = sorted({
    term.strip().lower()
    for term in [
        "musique", "concert", "culture", "spectacle", "théâtre", "theatre",
        "danse", "jeune public", "atelier", "exposition", "loisir", "art", "arts",
        "arts plastiques", "art contemporain", "conférence", "festival", "jazz", "cinéma",
        "cinema", "Conte", "enfant", "jeu vidéo", "jeux vidéo", "jeux vidéos", "jeux", "jeu",
        "projection", "numérique", "création", "poésie", "nature", "Activités et animations",
        "animation", "animations", "Histoire", "histoire", "patrimoine", "Fête de la science",
        "humour", "film", "live", "musée", "fête", "chant", "chanson", "ateliers", "Lecture",
        "science", "sciences", "Sciences et société", "visite", "performance", "photographie",
        "littérature", "médiathèque", "rock", "soirée", "jeunesse", "piano", "marionnettes",
        "découverte", "sortie", "documentaire", "débat", "musique classique", "manga",
        "environnement", "classique", "dessin", "contes", "stage", "écologie", "sculpture",
        "architecture", "electro", "spectacles", "écriture", "opéra", "opera",
        "Centre culturel canadien", "Montreuil", "ciné-club", "contemporain", "livre", "orchestre",
        "eSport", "e-sport", "pop", "archéologie", "rap", "balade", "vidéo", "éducation",
        "hip hop", "hip-hop", "comédie", "sens", "Luxembourg", "installation",
        "Conférences et débats", "philosophie", "artiste", "Centre d'art", "vernissage", "participatif",
        "échange", "concerts", "danse contemporaine", "métiers", "Atelier des sciences",
        "Centre d'art et de culture", "astronomie", "fablab", "Cirque", "philo", "photo",
    ]
    if term and term.strip()
})

IDF_REGION_TERMS = {"ile-de-france", "île-de-france"}
IDF_DEPARTMENT_CODES = {"75", "77", "78", "91", "92", "93", "94", "95"}

MIN_CONTENT_CHARS = 120


def parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = dateparser.parse(text)
        if dt is None:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def first_non_empty(record: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for key in keys:
        if key in record and record[key] not in (None, "", [], {}):
            return record[key]
    return None


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " | ".join(to_text(v) for v in value if v not in (None, ""))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def clean_text(value: str) -> str:
    # Normalise les espaces pour stabiliser la qualité des embeddings.
    return re.sub(r"\s+", " ", (value or "")).strip()


def strip_html(text: str) -> str:
    no_html = re.sub(r"<[^>]+>", " ", text or "")
    return clean_text(no_html)


def parse_timings_window(value: Any) -> tuple[Optional[datetime], Optional[datetime]]:
    if not value:
        return None, None

    payload = value
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return None, None

    if not isinstance(payload, list):
        return None, None

    starts: List[datetime] = []
    ends: List[datetime] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        begin_dt = parse_dt(item.get("begin"))
        end_dt = parse_dt(item.get("end"))
        if begin_dt:
            starts.append(begin_dt)
        if end_dt:
            ends.append(end_dt)

    return (min(starts) if starts else None, max(ends) if ends else None)


def sanitize_date_range(start_dt: Optional[datetime], end_dt: Optional[datetime]) -> tuple[Optional[datetime], Optional[datetime]]:
    if start_dt and end_dt and end_dt < start_dt:
        return end_dt, start_dt
    return start_dt, end_dt


def extract_tags(record: Dict[str, Any]) -> List[str]:
    keywords = first_non_empty(record, [
        "keywords", "keywords_fr", "tags", "theme", "themes", "themes_fr", "category"
    ])

    if isinstance(keywords, list):
        raw_tags = [to_text(x) for x in keywords if to_text(x)]
    elif keywords:
        raw_tags = [x.strip() for x in re.split(r"[;,|]", to_text(keywords)) if x.strip()]
    else:
        raw_tags = ["culture"]

    # Déduplique en conservant l'ordre.
    deduped: List[str] = []
    seen = set()
    for tag in raw_tags:
        normalized = clean_text(tag)
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized)

    return deduped or ["culture"]


def is_vectorizable(doc: Dict[str, Any]) -> bool:
    title = clean_text(to_text(doc.get("title")))
    content = clean_text(to_text(doc.get("content")))
    summary = clean_text(to_text(doc.get("summary")))

    if len(title) < 3:
        return False
    if len(content) < MIN_CONTENT_CHARS:
        return False
    # Un résumé absent est toléré si le contenu structuré est riche.
    if not summary and len(content) < (MIN_CONTENT_CHARS + 30):
        return False
    return True


def looks_cultural(record: Dict[str, Any]) -> bool:
    blob_parts = [
        to_text(first_non_empty(record, ["title", "title_fr", "name"])),
        to_text(first_non_empty(record, ["description", "description_fr", "longdescription", "lead_text", "keywords", "keywords_fr"])),
        to_text(first_non_empty(record, ["keywords", "keywords_fr", "tags", "theme", "themes", "themes_fr", "category", "origin_agenda_title"])),
    ]
    blob = strip_html(" ".join(blob_parts)).lower()
    return any(term in blob for term in CULTURE_TERMS)


def is_ile_de_france(record: Dict[str, Any]) -> bool:
    region_blob = clean_text(to_text(first_non_empty(record, [
        "location_region", "region", "region_name"
    ]))).lower()
    if any(term in region_blob for term in IDF_REGION_TERMS):
        return True

    department_blob = clean_text(to_text(first_non_empty(record, [
        "location_departmentcode", "departmentcode", "department_code", "location_department"
    ])))
    if department_blob:
        match = re.search(r"\b(" + "|".join(sorted(IDF_DEPARTMENT_CODES)) + r")\b", department_blob)
        if match:
            return True

    return False


def extract_dates(record: Dict[str, Any]) -> tuple[Optional[datetime], Optional[datetime]]:
    # Priorité aux champs fréquents OpenAgenda
    start = first_non_empty(record, [
        "firstdate_begin", "date_first", "date_start", "start", "starts_at",
        "begin", "daterange_start", "firstdate_end", "lastdate_begin"
    ])
    end = first_non_empty(record, [
        "lastdate_end", "date_last", "date_end", "end", "ends_at",
        "finish", "daterange_end", "lastdate_begin", "firstdate_end"
    ])

    start_dt = parse_dt(start)
    end_dt = parse_dt(end)

    # fallback : parfois il n'y a qu'une seule date
    if not start_dt:
        single = first_non_empty(record, ["date", "createdat", "updatedat"])
        start_dt = parse_dt(single)

    # fallback avancé : certaines sources fournissent surtout un tableau timings.
    if not start_dt or not end_dt:
        timings_start, timings_end = parse_timings_window(record.get("timings"))
        start_dt = start_dt or timings_start
        end_dt = end_dt or timings_end

    return sanitize_date_range(start_dt, end_dt)


def overlaps_window(
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
    window_start: datetime,
    window_end: datetime,
) -> bool:
    effective_start = start_dt or end_dt
    effective_end = end_dt or start_dt
    if not effective_start or not effective_end:
        return False

    # Garde tout événement dont l'intervalle croise la fenêtre [window_start, window_end].
    return effective_end >= window_start and effective_start <= window_end


def build_window_where(window_start: datetime, window_end: datetime) -> str:
    start_date = window_start.date().isoformat()
    end_date = window_end.date().isoformat()
    return (
        "location_countrycode='FR'"
        " AND (location_region='Île-de-France' OR location_region='Ile-de-France')"
        f" AND firstdate_begin <= date'{end_date}'"
        f" AND lastdate_end >= date'{start_date}'"
    )


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    title = clean_text(to_text(first_non_empty(record, ["title", "title_fr", "name"]))) or "Événement culturel"
    summary = strip_html(to_text(first_non_empty(record, ["description", "description_fr", "lead_text", "longdescription", "longdescription_fr"])))
    start_dt, end_dt = extract_dates(record)
    start_dt, end_dt = sanitize_date_range(start_dt, end_dt)

    city = clean_text(to_text(first_non_empty(record, ["location_city", "city"])))
    region_raw = clean_text(to_text(first_non_empty(record, ["location_region", "region"])))
    # Normalisation: Île-de-France -> Ile-de-France (graphie unique)
    region = region_raw.replace("Île-de-France", "Ile-de-France").replace("île-de-France", "Ile-de-France")
    country_code = clean_text(to_text(first_non_empty(record, ["location_countrycode", "countrycode"]))).upper() or "FR"
    location_name = clean_text(to_text(first_non_empty(record, ["location_name", "location", "address_name"])))
    address = clean_text(to_text(first_non_empty(record, ["location_address", "address"])))
    tags = extract_tags(record)

    record_id = clean_text(to_text(first_non_empty(record, ["uid", "slug", "id"]))) or title

    source_url = clean_text(to_text(first_non_empty(record, ["canonicalurl", "html", "url", "shareurl"])))
    if source_url and not source_url.startswith(("http://", "https://")):
        source_url = ""

    if not summary:
        summary = clean_text(to_text(first_non_empty(record, ["longdescription", "longdescription_fr", "description", "description_fr"])))
        summary = strip_html(summary)
    summary = summary[:1500]

    content_parts = [
        f"Titre: {title}",
        f"Résumé: {summary}" if summary else "",
        f"Début: {start_dt.isoformat()}" if start_dt else "",
        f"Fin: {end_dt.isoformat()}" if end_dt else "",
        f"Lieu: {location_name}" if location_name else "",
        f"Adresse: {address}" if address else "",
        f"Ville: {city}" if city else "",
        f"Région: {region}" if region else "",
        "Pays: France",
        f"Tags: {', '.join(tags)}" if tags else "",
    ]
    content = clean_text("\n".join(part for part in content_parts if part))

    missing_fields: List[str] = []
    if not summary:
        missing_fields.append("summary")
    if not city:
        missing_fields.append("city")
    if not region:
        missing_fields.append("region")
    if not start_dt:
        missing_fields.append("event_start")
    if not end_dt:
        missing_fields.append("event_end")

    return {
        "id": f"evenements-publics-openagenda:{record_id}",
        "title": title,
        "summary": summary,
        "content": content,
        "event_start": start_dt.isoformat() if start_dt else None,
        "event_end": end_dt.isoformat() if end_dt else None,
        "location_name": location_name or None,
        "address": address or None,
        "country": "France",
        "country_code": "FR" if country_code != "FR" else country_code,
        "city": city or None,
        "region": region or None,
        "tags": tags,
        "source_dataset": "evenements-publics-openagenda",
        "source_record_url": source_url or None,
        "source_platform": "public.opendatasoft.com",
        "metadata": {
            "raw_uid": to_text(first_non_empty(record, ["uid", "id", "slug"])),
            "updated_at": to_text(first_non_empty(record, ["updatedat", "updated_at"])),
            "quality_missing_fields": missing_fields,
        },
    }


def fetch_records(
    limit: int = 100,
    max_records: Optional[int] = None,
    where_clause: Optional[str] = None,
    max_offset: int = OPENAGENDA_MAX_OFFSET,
) -> Iterable[Dict[str, Any]]:
    offset = 0
    session = requests.Session()
    session.headers.update(HEADERS)

    while True:
        if offset >= max_offset:
            break

        remaining_offset = max_offset - offset
        if max_records is None:
            batch_limit = min(limit, remaining_offset)
        else:
            batch_limit = min(limit, max_records - offset, remaining_offset)

        if batch_limit <= 0:
            break

        params = {
            "lang": "fr",
            "limit": batch_limit,
            "offset": offset,
            "where": where_clause or "location_countrycode='FR'",
            "order_by": "lastdate_end desc",
        }
        response = session.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        if not results:
            break

        for rec in results:
            yield rec

        offset += len(results)
        if len(results) < batch_limit:
            break


def build_rag_file(
    output_path: str = "data/evenements_publics_openagenda_culture_ile_de_france_rag.jsonl",
    max_records: Optional[int] = None,
    max_offset: int = OPENAGENDA_MAX_OFFSET,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if max_offset > OPENAGENDA_MAX_OFFSET:
        max_offset = OPENAGENDA_MAX_OFFSET

    kept = 0
    fetched = 0
    skipped_non_cultural = 0
    skipped_outside_region = 0
    skipped_outside_window = 0
    skipped_not_vectorizable = 0
    skipped_duplicates = 0
    seen = set()
    where_clause = build_window_where(WINDOW_START_UTC, WINDOW_END_UTC)

    with output.open("w", encoding="utf-8") as f:
        for record in fetch_records(
            max_records=max_records,
            where_clause=where_clause,
            max_offset=max_offset,
        ):
            fetched += 1
            if not looks_cultural(record):
                skipped_non_cultural += 1
                continue

            if not is_ile_de_france(record):
                skipped_outside_region += 1
                continue

            start_dt, end_dt = extract_dates(record)
            if not overlaps_window(start_dt, end_dt, WINDOW_START_UTC, WINDOW_END_UTC):
                skipped_outside_window += 1
                continue

            doc = normalize_record(record)
            if not is_vectorizable(doc):
                skipped_not_vectorizable += 1
                continue

            dedup_key = doc["metadata"]["raw_uid"] or (doc["title"].strip().lower(), doc["event_start"], doc["city"])
            if dedup_key in seen:
                skipped_duplicates += 1
                continue
            seen.add(dedup_key)

            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            kept += 1

    print(
        f"{kept} événements écrits dans {output} sur {fetched} enregistrements FR/IDF analysés"
    )
    print(
        "Fenêtre forcée -> "
        f"{WINDOW_START_UTC.date().isoformat()} à {WINDOW_END_UTC.date().isoformat()}"
    )
    print(f"Offset OpenAgenda appliqué -> max_offset={max_offset} (plafond technique=10000)")
    print(
        "Rejets -> "
        f"non_culture={skipped_non_cultural}, "
        f"hors_idf={skipped_outside_region}, "
        f"hors_fenetre={skipped_outside_window}, "
        f"qualite_vecteur={skipped_not_vectorizable}, "
        f"doublons={skipped_duplicates}"
    )
    return output


if __name__ == "__main__":
    build_rag_file()
