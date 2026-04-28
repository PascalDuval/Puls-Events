import requests, json, re
CULTURE_TERMS = ["culture", "art", "arts", "musée", "musee", "musique", "concert", "théâtre", "theatre", "cinéma", "cinema", "festival", "exposition", "danse", "patrimoine", "spectacle", "bibliothèque", "bibliotheque", "lecture", "opéra", "opera", "dédicace", "dedicace", "atelier", "rencontre", "conférence", "conference"]
def to_text(v):
    if v is None: return ""
    if isinstance(v, list): return " | ".join(to_text(x) for x in v if x)
    return str(v).strip()
def looks_cultural(r):
    parts = [r.get("title_fr"), r.get("description_fr"), r.get("longdescription_fr"), r.get("keywords_fr"), r.get("originagenda_title")]
    blob = " ".join(to_text(p) for p in parts).lower()
    return any(t in blob for t in CULTURE_TERMS)
url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
res = requests.get(url, params={"where": "location_countrycode='FR'", "limit": 50}).json()
recs = res.get("results", [])
fields = ["uid", "title_fr", "firstdate_begin", "firstdate_end", "lastdate_begin", "lastdate_end", "updatedat", "originagenda_title", "keywords_fr"]
print("--- 10 Records ---")
for r in recs[:10]: print({f: r.get(f) for f in fields})
stats = {f: sum(1 for r in recs if r.get(f)) for f in fields}
cult_count = sum(1 for r in recs if looks_cultural(r))
print("\n--- Stats (50) ---")
for f, c in stats.items(): print(f"{f}: {c}")
print(f"Matches looks_cultural: {cult_count}")
print("\n--- Not cultural but maybe ---")
maybe = [r for r in recs if not looks_cultural(r) and any(w in (to_text(r.get("originagenda_title"))).lower() for w in ["loisir", "animation", "fête", "fete", "sport", "social", "jeunesse", "jeune"])]
for r in maybe[:5]: print(f"UID: {r.get('uid')} | Title: {r.get('title_fr')} | Agenda: {r.get('originagenda_title')} | Keywords: {r.get('keywords_fr')}")
