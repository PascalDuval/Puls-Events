import requests


def test_api() -> None:
    url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records/?limit=3"

    response = requests.get(url, timeout=30)
    print(response.status_code)
    print(response.json())

    assert response.status_code == 200
