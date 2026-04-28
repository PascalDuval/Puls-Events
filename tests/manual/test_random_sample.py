import json
import random
from pathlib import Path


def test_random_sample():
    """Charge le fichier JSONL et affiche 3 événements aléatoires avec les bons champs."""
    project_root = Path(__file__).resolve().parents[2]
    jsonl_path = project_root / "data" / "evenements_publics_openagenda_culture_ile_de_france_rag.jsonl"

    if not jsonl_path.exists():
        print(f"ERROR Fichier JSONL non trouvé: {jsonl_path}")
        return

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        print("ERROR Aucun record trouvé dans le JSONL")
        return

    print(f"OK Total d'événements: {len(records)}")

    sample = random.sample(records, min(3, len(records)))

    for i, record in enumerate(sample, 1):
        print(f"\n{'='*80}")
        print(f"EVENT Événement #{i}")
        print(json.dumps(record, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    test_random_sample()
