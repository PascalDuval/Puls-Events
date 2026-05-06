from __future__ import annotations

import os
from pathlib import Path


def _load_env_value(path: Path, key: str) -> str:
    if not path.exists() or not path.is_file():
        return ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip('"').strip("'")
    return ""


def get_mistral_api_key() -> str:
    # Priorite: variable d'environnement puis fichier .env local du projet.
    value = os.getenv("MISTRAL_API_KEY", "").strip()
    if value:
        return value
    project_env = Path(__file__).resolve().parents[1] / ".env"
    return _load_env_value(project_env, "MISTRAL_API_KEY")


MISTRAL_API_KEY = get_mistral_api_key()