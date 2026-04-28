from __future__ import annotations

from datetime import datetime

from utils.temporal_deixis import PARIS_TZ, infer_temporal_window


def test_infer_ce_weekend_from_friday() -> None:
    now_fr = datetime(2026, 4, 24, 10, 30, tzinfo=PARIS_TZ)  # vendredi
    window = infer_temporal_window("Quels concerts sur Paris ce week-end ?", now_fr=now_fr)

    assert window is not None
    assert window.label == "ce_weekend"
    assert window.after_date_utc.startswith("2026-04-24")
    assert window.before_date_utc.startswith("2026-04-26")


def test_infer_ce_soir() -> None:
    now_fr = datetime(2026, 4, 23, 14, 0, tzinfo=PARIS_TZ)
    window = infer_temporal_window("Qu'y a-t-il ce soir a Lyon ?", now_fr=now_fr)

    assert window is not None
    assert window.label == "ce_soir"
    assert "T16:00:00+00:00" in window.after_date_utc or "T17:00:00+00:00" in window.after_date_utc


def test_do_not_infer_complex_non_deictic_month_window() -> None:
    now_fr = datetime(2026, 4, 23, 9, 0, tzinfo=PARIS_TZ)
    window = infer_temporal_window(
        "Tous les concerts a venir jusqu'a fin mai 2026", now_fr=now_fr
    )

    assert window is None


def test_infer_demain_soir() -> None:
    now_fr = datetime(2026, 4, 23, 9, 0, tzinfo=PARIS_TZ)
    window = infer_temporal_window("Quels concerts demain soir a Paris ?", now_fr=now_fr)

    assert window is not None
    assert window.label == "demain_soir"
    # 18:00 Europe/Paris => 16:00 UTC fin avril (heure d'ete)
    assert window.after_date_utc.startswith("2026-04-24T16:00:00")


def test_infer_apres_demain_soir() -> None:
    now_fr = datetime(2026, 4, 23, 9, 0, tzinfo=PARIS_TZ)
    window = infer_temporal_window("Quels concerts apres-demain soir ?", now_fr=now_fr)

    assert window is not None
    assert window.label == "apres_demain_soir"
    assert window.after_date_utc.startswith("2026-04-25T16:00:00")


def test_infer_none_when_no_temporal_phrase() -> None:
    now_fr = datetime(2026, 4, 23, 9, 0, tzinfo=PARIS_TZ)
    window = infer_temporal_window("Propose des concerts de jazz", now_fr=now_fr)

    assert window is None
