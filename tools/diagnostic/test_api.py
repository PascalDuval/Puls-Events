import requests
import json

base_url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"

queries = [
    "location_countrycode='FR' AND lastdate_end >= date'2025-04-22'",
    "location_countrycode='FR' AND firstdate_begin <= date'2026-04-22' AND lastdate_end >= date'2025-04-22'",
    "location_countrycode='FR' AND firstdate_begin <= '2026-04-22' AND lastdate_end >= '2025-04-22'"
]

for i, q in enumerate(queries, 1):
    print(f"--- Appel {i} ---")
    params = {"where": q, "limit": 1}
    r = requests.get(base_url, params=params)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Total Count: {data.get('total_count')}")
        results = data.get("results", [])
        if results:
            res = results[0]
            print(f"Record: uid={res.get('uid')}, title_fr={res.get('title_fr')}, firstdate_begin={res.get('firstdate_begin')}, lastdate_end={res.get('lastdate_end')}")
    else:
        print(f"Error: {r.text}")
