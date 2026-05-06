from __future__ import annotations

import os
from pathlib import Path


def _is_placeholder_key(value: str) -> bool:
    if not value:
        return True
    normalized = value.strip().lower()
    known_placeholders = {
        "your_mistral_api_key_here",
        "changeme",
        "change_me",
        "replace_me",
        "none",
        "null",
    }
    if normalized in known_placeholders:
        return True
    # Cle de demo observee dans des exercices: ne pas l'utiliser en production.
    if normalized.startswith("x75slf"):
        return True
    return False


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
    if value and not _is_placeholder_key(value):
        return value

    current = Path(__file__).resolve()
    env_candidates = [
        current.parents[1] / ".env",  # Pull-Events/.env
        current.parents[2] / ".env",  # racine du workspace
    ]
    for env_path in env_candidates:
        file_value = _load_env_value(env_path, "MISTRAL_API_KEY")
        if file_value and not _is_placeholder_key(file_value):
            return file_value
    return ""


MISTRAL_API_KEY = get_mistral_api_key()