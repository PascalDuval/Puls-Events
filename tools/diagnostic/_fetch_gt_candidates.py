"""
Script temporaire : collecte des candidats d'événements réels sur OpenDataSoft
pour construire la ground truth RAGAS.
Fenêtre : 2025-04-25 → 2027-04-25.
Résultats écrits dans tools/diagnostic/_gt_candidates_raw.json (à valider).
"""

import json
import time
import urllib.parse
from pathlib import Path

import requests

BASE_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets"
    "/evenements-publics-openagenda/records/"
)

# Fenêtre stricte : l'événement COMMENCE entre 2025-04-25 et 2027-04-25
DATE_FILTER = (
    "firstdate_begin >= date'2025-04-25'"
    " AND firstdate_begin <= date'2027-04-25'"
)

IDF_FILTER = "(location_region='Île-de-France' OR location_region='Ile-de-France')"

# Recherche textuelle dans title_fr, description_fr et keywords_fr
def txt(field: str, kw: str) -> str:
    return f"{field} like '%{kw}%'"

def any_field(kws: list[str]) -> str:
    """Cherche les mots-clés dans title_fr, description_fr et keywords_fr."""
    conditions = []
    for kw in kws:
        for field in ("title_fr", "description_fr", "keywords_fr"):
            conditions.append(txt(field, kw))
    return "(" + " OR ".join(conditions) + ")"


QUERIES = [
    {
        "id": "Q1",
        "label": "Concert de jazz en Île-de-France",
        "question": "as-tu un concert de jazz en Île-de-France ?",
        "where": f"{any_field(['jazz'])} AND {IDF_FILTER} AND {DATE_FILTER}",
    },
    {
        "id": "Q2",
        "label": "Exposition photo en Île-de-France",
        "question": "je cherche une exposition photo en Île-de-France",
        "where": f"{any_field(['photo', 'photographie'])} AND {IDF_FILTER} AND {DATE_FILTER}",
    },
    {
        "id": "Q3",
        "label": "Sortie famille / enfants en Île-de-France",
        "question": "quelle sortie pour une famille avec enfants en Île-de-France ?",
        "where": f"{any_field(['famille', 'enfant'])} AND {IDF_FILTER} AND {DATE_FILTER}",
    },
    {
        "id": "Q4",
        "label": "Sciences / astronomie en Île-de-France",
        "question": "propose-moi une activité autour des sciences ou de l'astronomie",
        "where": f"{any_field(['astronomie', 'planétarium', 'sciences'])} AND {IDF_FILTER} AND {DATE_FILTER}",
    },
    {
        "id": "Q5",
        "label": "Événements à Aubervilliers",
        "question": "propose-moi un spectacle à Aubervilliers",
        "where": f"location_city='Aubervilliers' AND {DATE_FILTER}",
    },
    {
        "id": "Q6",
        "label": "Activité pour adolescents en Île-de-France",
        "question": "propose-moi une activité gratuite pour adolescents en Île-de-France",
        "where": f"{any_field(['ados', 'adolescent', 'jeune'])} AND {IDF_FILTER} AND {DATE_FILTER}",
    },
    {
        "id": "Q7",
        "label": "Atelier créatif / artistique en Île-de-France",
        "question": "y a-t-il un atelier créatif ou artistique pour adultes en Île-de-France ?",
        "where": f"{any_field(['atelier'])} AND {IDF_FILTER} AND {DATE_FILTER}",
    },
    {
        "id": "Q8",
        "label": "Danse contemporaine en Île-de-France",
        "question": "je cherche un événement autour de la danse contemporaine",
        "where": f"{any_field(['danse', 'contemporaine', 'contemporain'])} AND {IDF_FILTER} AND {DATE_FILTER}",
    },
]

SELECT_FIELDS = "title_fr,location_city,firstdate_begin,lastdate_end,keywords_fr,canonicalurl"


def fetch_events(where: str, limit: int = 6) -> list[dict]:
    params = {
        "where": where,
        "select": SELECT_FIELDS,
        "limit": limit,
        "lang": "fr",
        "timezone": "UTC",
    }
    url = BASE_URL + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", [])


def main() -> None:
    output: list[dict] = []

    for q in QUERIES:
        print(f"[{q['id']}] {q['label']} ...")
        try:
            events = fetch_events(q["where"])
        except Exception as exc:
            print(f"  ERREUR: {exc}")
            events = []

        print(f"  → {len(events)} événements retournés")
        for e in events:
            print(
                f"    • {e.get('title_fr', '(sans titre)')} | "
                f"{e.get('location_city', '')} | "
                f"{str(e.get('firstdate_begin', ''))[:10]} → {str(e.get('lastdate_end', ''))[:10]} | "
                f"{e.get('canonicalurl', '')}"
            )

        output.append({
            "id": q["id"],
            "label": q["label"],
            "question": q["question"],
            "events": events,
        })
        time.sleep(0.5)  # politesse API

    out_path = Path(__file__).parent / "_gt_candidates_raw.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultats écrits dans : {out_path}")


if __name__ == "__main__":
    main()
