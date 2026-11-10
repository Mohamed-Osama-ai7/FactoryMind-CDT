"""Command-line entry points for the MetroPT-3 pipeline.

uv run python -m factorymind.anomaly.cli prepare    # download, verify, validate, store Parquet
uv run python -m factorymind.anomaly.cli train      # run experiment, save results and models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from factorymind.anomaly import experiment
from factorymind.config import get_settings
from factorymind.data import metropt3

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "metropt3_anomaly.json"


def _summary(results: dict[str, Any]) -> str:
    lines = [f"version {results['version']}  selected: {results['selected_detector']}"]
    detectors: dict[str, Any] = results["detectors"]
    header = f"{'detector':<18}{'split':<11}{'events':>8}{'FA/day':>9}{'AP':>8}{'q':>7}{'k':>3}"
    lines.append(header)
    for name, det in detectors.items():
        for split in ("validation", "test"):
            r = det[split]
            ap = r["average_precision"]
            lines.append(
                f"{name:<18}{split:<11}{r['events_detected']:>4}/{r['events_total']:<3}"
                f"{r['false_alarms_per_day']:>9.3f}{(ap if ap is not None else float('nan')):>8.3f}"
                f"{det['threshold_quantile']:>7}{det['persistence']:>3}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="factorymind.anomaly.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="download, verify and validate MetroPT-3")
    train = sub.add_parser("train", help="run the experiment and save models")
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.command == "prepare":
        manifest = metropt3.prepare(settings.raw_dir, settings.processed_dir)
        print(json.dumps(manifest["raw_quality"], indent=2))
        return 0

    config = experiment.ExperimentConfig.load(args.config)
    telemetry = metropt3.load_processed(settings.processed_dir)
    results, bundles, scored = experiment.run(config, telemetry)
    run_dir = experiment.save(
        results, bundles, scored, settings.artifacts_dir, settings.anomaly_model_name
    )
    print(_summary(results))
    print(f"results: {run_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
