from __future__ import annotations

import calendar
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo


PARIS_TZ = ZoneInfo("Europe/Paris")

MONTHS_FR = {
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "aout": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12,
}

_MONTHS_PATTERN = "(" + "|".join(MONTHS_FR.keys()) + ")"


@dataclass
class TemporalWindow:
    after_date_utc: str
    before_date_utc: str
    label: str


def normalize_fr(text: str) -> str:
    text = text or ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _to_utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _day_bounds(day: datetime) -> tuple[datetime, datetime]:
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    end = day.replace(hour=23, minute=59, second=59, microsecond=0)
    return start, end


def _evening_bounds(day: datetime) -> tuple[datetime, datetime]:
    start = day.replace(hour=18, minute=0, second=0, microsecond=0)
    end = day.replace(hour=23, minute=59, second=59, microsecond=0)
    return start, end


def _weekend_bounds(now_fr: datetime, next_weekend: bool) -> tuple[datetime, datetime]:
    weekday = now_fr.weekday()
    if next_weekend:
        days_until_saturday = (5 - weekday) % 7
        if days_until_saturday == 0:
            days_until_saturday = 7
    else:
        if weekday <= 4:
            days_until_saturday = 5 - weekday
        else:
            days_until_saturday = -(weekday - 5)

    saturday = (now_fr + timedelta(days=days_until_saturday)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    sunday_end = (saturday + timedelta(days=1)).replace(
        hour=23, minute=59, second=59, microsecond=0
    )
    return saturday, sunday_end


def infer_temporal_window(
    question: str,
    now_fr: Optional[datetime] = None,
    tz: ZoneInfo = PARIS_TZ,
) -> Optional[TemporalWindow]:
    """Infer a UTC temporal window from French deictic expressions."""
    now_fr = now_fr or datetime.now(tz)
    text = normalize_fr(question)

    # 1) Ce week-end / week-end prochain
    if re.search(r"\bce\s+week[ -]?end\b|\bce\s+weekend\b", text):
        start, end = _weekend_bounds(now_fr, next_weekend=False)
        return TemporalWindow(
            after_date_utc=_to_utc_iso(start),
            before_date_utc=_to_utc_iso(end),
            label="ce_weekend",
        )

    if re.search(r"\bweek[ -]?end\s+prochain\b|\bweekend\s+prochain\b", text):
        start, end = _weekend_bounds(now_fr, next_weekend=True)
        return TemporalWindow(
            after_date_utc=_to_utc_iso(start),
            before_date_utc=_to_utc_iso(end),
            label="weekend_prochain",
        )

    # 2) Aujourd'hui / demain / apres-demain (+ variantes soir)
    if re.search(r"\bapres[- ]demain\s+soir\b", text):
        ref = now_fr + timedelta(days=2)
        start, end = _evening_bounds(ref)
        return TemporalWindow(
            after_date_utc=_to_utc_iso(start),
            before_date_utc=_to_utc_iso(end),
            label="apres_demain_soir",
        )

    if re.search(r"\bdemain\s+soir\b", text):
        ref = now_fr + timedelta(days=1)
        start, end = _evening_bounds(ref)
        return TemporalWindow(
            after_date_utc=_to_utc_iso(start),
            before_date_utc=_to_utc_iso(end),
            label="demain_soir",
        )

    if re.search(r"\baujourd[' ]hui\b", text):
        start, end = _day_bounds(now_fr)
        return TemporalWindow(
            after_date_utc=_to_utc_iso(start),
            before_date_utc=_to_utc_iso(end),
            label="aujourdhui",
        )

    if re.search(r"\bapres[- ]demain\b", text):
        ref = now_fr + timedelta(days=2)
        start, end = _day_bounds(ref)
        return TemporalWindow(
            after_date_utc=_to_utc_iso(start),
            before_date_utc=_to_utc_iso(end),
            label="apres_demain",
        )

    if re.search(r"\bdemain\b", text):
        ref = now_fr + timedelta(days=1)
        start, end = _day_bounds(ref)
        return TemporalWindow(
            after_date_utc=_to_utc_iso(start),
            before_date_utc=_to_utc_iso(end),
            label="demain",
        )

    # 3) Ce soir
    if re.search(r"\bce\s+soir\b", text):
        start, end = _evening_bounds(now_fr)
        if now_fr > end:
            next_day = now_fr + timedelta(days=1)
            start, end = _evening_bounds(next_day)
        return TemporalWindow(
            after_date_utc=_to_utc_iso(start),
            before_date_utc=_to_utc_iso(end),
            label="ce_soir",
        )

    # 4) Cette semaine / semaine prochaine
    if re.search(r"\bcette\s+semaine\b", text):
        monday = (now_fr - timedelta(days=now_fr.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        sunday_end = (monday + timedelta(days=6)).replace(
            hour=23, minute=59, second=59, microsecond=0
        )
        return TemporalWindow(
            after_date_utc=_to_utc_iso(monday),
            before_date_utc=_to_utc_iso(sunday_end),
            label="cette_semaine",
        )

    if re.search(r"\bsemaine\s+prochaine\b", text):
        monday_next = (now_fr - timedelta(days=now_fr.weekday()) + timedelta(days=7)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        sunday_end = (monday_next + timedelta(days=6)).replace(
            hour=23, minute=59, second=59, microsecond=0
        )
        return TemporalWindow(
            after_date_utc=_to_utc_iso(monday_next),
            before_date_utc=_to_utc_iso(sunday_end),
            label="semaine_prochaine",
        )

    # 5) Mois explicite : "en mai", "en mai 2026", "de mai 2026", "pour juin", "au mois de mars"
    _month_re = (
        r"\b(?:en|de|du|pour|pendant|au\s+mois\s+de)\s+"
        + _MONTHS_PATTERN
        + r"\s*(\d{4})?\b"
    )
    m = re.search(_month_re, text)
    if m:
        month_name = m.group(1)
        year_str = m.group(2)
        month_num = MONTHS_FR[month_name]
        if year_str:
            year = int(year_str)
        else:
            # Si le mois est déjà passé cette année, prendre l'année prochaine
            year = now_fr.year
            if month_num < now_fr.month:
                year += 1
        _, last_day = calendar.monthrange(year, month_num)
        start = now_fr.replace(
            year=year, month=month_num, day=1,
            hour=0, minute=0, second=0, microsecond=0
        )
        end = now_fr.replace(
            year=year, month=month_num, day=last_day,
            hour=23, minute=59, second=59, microsecond=0
        )
        return TemporalWindow(
            after_date_utc=_to_utc_iso(start),
            before_date_utc=_to_utc_iso(end),
            label=f"{month_name}_{year}",
        )

    # 6) Saisons : "en été", "l'été", "cet été", "au printemps", "en automne", "en hiver" (+ année optionnelle)
    # Bornes météo : printemps=mars-mai, été=juin-août, automne=sept-nov, hiver=déc-fév
    _SEASONS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
        "ete":       ((6, 1), (8, 31)),
        "printemps": ((3, 1), (5, 31)),
        "automne":   ((9, 1), (11, 30)),
        "hiver":     ((12, 1), (2, 28)),
    }
    _season_re = r"\b(?:en|l['\s]|cet|au|a\s+l[']\s*)\s*(?:ete|printemps|automne|hiver)\b"
    _season_name_re = r"(?:ete|printemps|automne|hiver)"
    _season_match = re.search(
        r"\b(?:en|l['\s]|cet|au|a\s+l['\s])\s*(" + _season_name_re + r")\s*(\d{4})?",
        text,
    )
    if _season_match:
        season_key = _season_match.group(1)
        year_str = _season_match.group(2)
        if season_key in _SEASONS:
            (sm, sd), (em, ed) = _SEASONS[season_key]
            year = int(year_str) if year_str else now_fr.year
            # Pour l'hiver, la fin est en février de l'année suivante
            end_year = year + 1 if season_key == "hiver" else year
            if ed == 28:
                _, ed = calendar.monthrange(end_year, em)
            start = now_fr.replace(year=year, month=sm, day=sd,
                                   hour=0, minute=0, second=0, microsecond=0)
            end = now_fr.replace(year=end_year, month=em, day=ed,
                                 hour=23, minute=59, second=59, microsecond=0)
            return TemporalWindow(
                after_date_utc=_to_utc_iso(start),
                before_date_utc=_to_utc_iso(end),
                label=f"{season_key}_{year}",
            )

    return None
