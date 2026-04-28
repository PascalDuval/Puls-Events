import requests, json
url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
# Request localized content if possible or just check what the structure is
res = requests.get(url, params={"where": "location_countrycode='FR'", "limit": 1}).json()
print(json.dumps(res.get("results"), indent=2))
