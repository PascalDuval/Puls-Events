import requests
import json
import re

CULTURE_TERMS = [
    'culture', 'art', 'arts', 'musée', 'musee', 'musique', 'concert',
    'théâtre', 'theatre', 'cinéma', 'cinema', 'festival', 'exposition',
    'danse', 'patrimoine', 'spectacle', 'bibliothèque', 'bibliotheque',
    'lecture', 'opéra', 'opera', 'dédicace', 'dedicace', 'atelier',
    'rencontre', 'conférence', 'conference'
]

def to_text(value):
    if value is None: return ''
    if isinstance(value, list): return ' | '.join(to_text(v) for v in value if v)
    return str(value).strip()

def looks_cultural(record):
    blob_parts = [
        to_text(record.get('title')),
        to_text(record.get('description')),
        to_text(record.get('keywords')),
        to_text(record.get('theme')),
        to_text(record.get('themes')),
        to_text(record.get('origin_agenda_title'))
    ]
    blob = ' '.join(blob_parts).lower()
    return any(term in blob for term in CULTURE_TERMS)

url = 'https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records'
params = {'where': \"location_countrycode='FR'\", 'limit': 50}
r = requests.get(url, params=params).json()
records = r.get('results', [])

fields_to_check = [
    'uid', 'title', 'firstdate_begin', 'firstdate_end', 'lastdate_begin', 
    'lastdate_end', 'date_first', 'date_last', 'keywords', 'theme', 'themes', 'origin_agenda_title'
]

print(f'--- 10 Records Detail ---')
for rec in records[:10]:
    detail = {f: rec.get(f) for f in fields_to_check}
    print(json.dumps(detail, ensure_ascii=False))

stats = {f: 0 for f in fields_to_check}
cultural_count = 0
not_cultural_but_maybe = []

for rec in records:
    for f in fields_to_check:
        if rec.get(f):
            stats[f] += 1
    
    is_cult = looks_cultural(rec)
    if is_cult:
        cultural_count += 1
    else:
        # Simple heuristic for 'maybe' cultural if it failed the strict check but has interesting agenda/theme
        maybe_blob = (to_text(rec.get('origin_agenda_title')) + ' ' + to_text(rec.get('theme')) + ' ' + to_text(rec.get('themes'))).lower()
        if any(w in maybe_blob for w in ['loisir', 'animation', 'fête', 'fete', 'sport', 'social', 'jeunesse']):
             if len(not_cultural_but_maybe) < 5:
                 not_cultural_but_maybe.append(rec)

print(f'\n--- Statistics (on 50 records) ---')
for f, count in stats.items():
    print(f'{f}: {count}')
print(f'Matches looks_cultural: {cultural_count}')

print(f'\n--- 5 Potential cultural records NOT matching looks_cultural ---')
for rec in not_cultural_but_maybe:
     print(f\"UID: {rec.get('uid')} | Title: {rec.get('title')} | Agenda: {rec.get('origin_agenda_title')} | Theme: {rec.get('theme')}\")

