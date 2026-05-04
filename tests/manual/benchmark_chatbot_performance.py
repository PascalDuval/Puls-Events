from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_chatbot_mistral import MistralRAGChatbot


def _split_csv(value: Optional[str], cast=str) -> List[Any]:
    if not value:
        return []
    out: List[Any] = []
    for part in value.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(cast(p))
    return out


def _compute_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "stdev_ms": 0.0,
            "p95_ms": 0.0,
        }

    vals = sorted(values)
    if len(vals) == 1:
        p95 = vals[0]
    else:
        index = int(round(0.95 * (len(vals) - 1)))
        p95 = vals[index]

    return {
        "count": float(len(vals)),
        "min_ms": min(vals),
        "max_ms": max(vals),
        "mean_ms": statistics.mean(vals),
        "median_ms": statistics.median(vals),
        "stdev_ms": statistics.stdev(vals) if len(vals) >= 2 else 0.0,
        "p95_ms": p95,
    }


@dataclass
class Scenario:
    name: str
    question: str
    k: int
    city: Optional[str] = None
    region: Optional[str] = None
    tags: Optional[List[str]] = None
    after_date: Optional[str] = None
    before_date: Optional[str] = None


def _build_scenarios(k_values: List[int], filter_mode: str) -> List[Scenario]:
    if not k_values:
        k_values = [6]

    use_filters = filter_mode.lower() == "strict"

    scenarios: List[Scenario] = []
    base_questions = [
        (
            "concert_jazz_idf",
            "Peux-tu me proposer des concerts de jazz en Ile-de-France ?",
            "Paris",
            ["jazz", "concert"],
        ),
        (
            "expo_photo_idf",
            "Je cherche des expositions photo interessantes en Ile-de-France.",
            "Versailles",
            ["exposition", "photo"],
        ),
        (
            "sorties_famille_idf",
            "Quelles sorties culturelles pour une famille me recommandes-tu en Ile-de-France ?",
            "Montreuil",
            ["famille", "enfant"],
        ),
    ]

    for k in k_values:
        for key, question, city, tags in base_questions:
            scenarios.append(
                Scenario(
                    name=f"{key}_k{k}",
                    question=question,
                    k=k,
                    city=city if use_filters else None,
                    region="Ile-de-France" if use_filters else None,
                    tags=tags if use_filters else None,
                )
            )

    # Scenario volontairement plus large pour evaluer le comportement guardrail.
    scenarios.append(
        Scenario(
            name="broad_query_guardrail",
            question="Donne-moi tous les evenements culturels disponibles en Ile-de-France.",
            k=max(k_values),
            region="Ile-de-France" if use_filters else None,
        )
    )

    return scenarios


def run_benchmark(rounds: int, k_values: List[int], warmup: int, pause_ms: int, filter_mode: str) -> Dict[str, Any]:
    chatbot = MistralRAGChatbot(index_dir="data")
    scenarios = _build_scenarios(k_values=k_values, filter_mode=filter_mode)

    if warmup > 0:
        for _ in range(warmup):
            chatbot.ask("Peux-tu recommander un evenement culturel en Ile-de-France ?", k=max(k_values) if k_values else 6)

    all_latencies: List[float] = []
    per_scenario_results: List[Dict[str, Any]] = []

    for scenario in scenarios:
        latencies_ms: List[float] = []
        docs_count: List[int] = []
        answer_lengths: List[int] = []
        errors: List[str] = []

        for _round in range(rounds):
            start = time.perf_counter()
            try:
                answer = chatbot.ask(
                    question=scenario.question,
                    k=scenario.k,
                    city=scenario.city,
                    region=scenario.region,
                    tags=scenario.tags,
                    after_date=scenario.after_date,
                    before_date=scenario.before_date,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                latencies_ms.append(elapsed_ms)
                all_latencies.append(elapsed_ms)
                docs_count.append(len(answer.documents))
                answer_lengths.append(len(answer.answer or ""))
            except Exception as exc:  # pragma: no cover - instrumentation runtime
                errors.append(str(exc))

            if pause_ms > 0:
                time.sleep(pause_ms / 1000.0)

        stats = _compute_stats(latencies_ms)
        per_scenario_results.append(
            {
                "scenario": scenario.name,
                "question": scenario.question,
                "params": {
                    "k": scenario.k,
                    "city": scenario.city,
                    "region": scenario.region,
                    "tags": scenario.tags,
                    "after_date": scenario.after_date,
                    "before_date": scenario.before_date,
                },
                "latency": stats,
                "avg_docs_retrieved": statistics.mean(docs_count) if docs_count else 0.0,
                "hit_rate_docs_gt0": (sum(1 for v in docs_count if v > 0) / len(docs_count)) if docs_count else 0.0,
                "avg_answer_length": statistics.mean(answer_lengths) if answer_lengths else 0.0,
                "errors": errors,
            }
        )

    successful_rounds = 0
    rounds_with_docs = 0
    for scenario_item in per_scenario_results:
        scenario_rounds = int(scenario_item["latency"]["count"])
        successful_rounds += scenario_rounds
        rounds_with_docs += int(round(scenario_item["hit_rate_docs_gt0"] * scenario_rounds))

    global_stats = _compute_stats(all_latencies)
    return {
        "rounds": rounds,
        "warmup": warmup,
        "k_values": k_values,
        "filter_mode": filter_mode,
        "scenario_count": len(scenarios),
        "global_latency": global_stats,
        "global_hit_rate_docs_gt0": (rounds_with_docs / successful_rounds) if successful_rounds else 0.0,
        "scenarios": per_scenario_results,
    }


def _print_human_report(report: Dict[str, Any]) -> None:
    print("=== Benchmark performance chatbot RAG ===")
    print(
        f"Rounds={report['rounds']} | Warmup={report['warmup']} | "
        f"Scenarios={report['scenario_count']} | K={report['k_values']} | filter_mode={report['filter_mode']}"
    )

    gl = report["global_latency"]
    print(
        "Global latency (ms): "
        f"mean={gl['mean_ms']:.2f} | median={gl['median_ms']:.2f} | "
        f"p95={gl['p95_ms']:.2f} | min={gl['min_ms']:.2f} | max={gl['max_ms']:.2f}"
    )
    print(f"Global hit-rate docs>0: {report['global_hit_rate_docs_gt0'] * 100:.1f}%")

    print("\n--- Details par scenario ---")
    for item in report["scenarios"]:
        lt = item["latency"]
        print(
            f"- {item['scenario']}: mean={lt['mean_ms']:.2f}ms, "
            f"median={lt['median_ms']:.2f}ms, p95={lt['p95_ms']:.2f}ms, "
            f"avg_docs={item['avg_docs_retrieved']:.2f}, "
            f"hit_rate_docs>0={item['hit_rate_docs_gt0'] * 100:.1f}%, "
            f"errors={len(item['errors'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark de rapidite d'acces a l'information via le chatbot RAG."
    )
    parser.add_argument("--rounds", type=int, default=3, help="Nombre de rounds par scenario")
    parser.add_argument("--k-values", type=str, default="5,10", help="Valeurs de K, CSV")
    parser.add_argument("--warmup", type=int, default=1, help="Nombre d'iterations de warmup")
    parser.add_argument(
        "--pause-ms",
        type=int,
        default=250,
        help="Pause entre deux requetes pour lisser les pics API",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Chemin optionnel pour sauvegarder le rapport JSON",
    )
    parser.add_argument(
        "--filter-mode",
        type=str,
        choices=["strict", "relaxed"],
        default="relaxed",
        help="strict=ville/region/tags actives, relaxed=sans filtres metadata",
    )
    args = parser.parse_args()

    k_values = _split_csv(args.k_values, cast=int)
    report = run_benchmark(
        rounds=args.rounds,
        k_values=k_values,
        warmup=args.warmup,
        pause_ms=args.pause_ms,
        filter_mode=args.filter_mode,
    )

    _print_human_report(report)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nRapport JSON ecrit: {output_path}")


if __name__ == "__main__":
    main()