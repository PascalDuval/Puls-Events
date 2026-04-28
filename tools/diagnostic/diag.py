import openagenda_culture_france_rag as mod
import requests
from datetime import datetime, timedelta, timezone

def diag():
    recs = []
    # fetch_records est un générateur, donc on itère directly
    count = 0
    for r in mod.fetch_records(limit=100, max_records=1000):
        recs.append(r)
        count += 1
        if count >= 1000: break
    
    cultural = [r for r in recs if mod.looks_cultural(r)]
    
    # Check individual filters in is_recent
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=365)
    
    recent = []
    rej_dates = []
    for r in cultural:
        sd, ed = mod.extract_dates(r)
        if mod.is_recent(sd, ed, 365):
            recent.append(r)
        else:
            if len(rej_dates) < 5:
                rej_dates.append({"t": r.get("title"), "sd": sd.isoformat() if sd else None, "ed": ed.isoformat() if ed else None})

    # Dedup check
    seen = set()
    unique = []
    for r in recent:
        sd, ed = mod.extract_dates(r)
        key = (str(r.get("title")).lower().strip(), sd.isoformat() if sd else None, str(r.get("location_city")).lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print(f"Sample: {len(recs)}")
    print(f"Pass Cultural: {len(cultural)}")
    print(f"Pass Recent: {len(recent)}")
    print(f"Unique: {len(unique)}")
    print("Rejected dates examples (Cultural but not Recent):")
    for d in rej_dates: print(f"  - {d}")

    # API counts
    base_url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
    try:
        r_fr = requests.get(base_url, params={"where": "location_countrycode='FR'", "limit": 0}).json().get("total_count")
        r_24 = requests.get(base_url, params={"where": "location_countrycode='FR' AND updatedat >= date'2024-01-01'", "limit": 0}).json().get("total_count")
        print(f"API Total FR: {r_fr}")
        print(f"API Total 2024+: {r_24}")
    except: pass

diag()
