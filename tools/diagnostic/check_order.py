import requests
base_url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
r = requests.get(base_url, params={"limit": 5, "where": "location_countrycode='FR'"}).json()
for i, res in enumerate(r.get("results", [])):
    print(f"Record {i}: updatedat={res.get('updatedat')}, firstdate_begin={res.get('firstdate_begin')}, lastdate_end={res.get('lastdate_end')}")
