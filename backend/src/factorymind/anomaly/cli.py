"""Command-line entry points for the MetroPT-3 pipeline.

uv run python -m factorymind.anomaly.cli prepare    # download, verify, validate, store Parquet
uv run python -m factorymind.anomaly.cli train      # run the protocol, save results and models
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
    lines = [
        f"version {results['version']}  default: {results['selected_detector']} "
        f"({results['selection_rationale']})",
        f"{'detector':<17}{'LOEO leaks':>11}{'FA/day':>8}{'opFA/d':>8}{'|delay|':>9}"
        f"{'faults':>9}{'ctrl':>7}{'clean':>8}{'lift':>7}{'p<=a':>8}",
    ]
    detectors: dict[str, Any] = results["detectors"]
    for name, d in detectors.items():
        loeo = d["loeo"]
        faults = d["fault_injection"]
        delay = loeo["median_abs_delay_minutes"]
        clean = (
            f"{faults['clean_control_detected']}/{faults['clean_control_cases']}" if faults else "-"
        )
        lift = (
            f"{faults['incremental_lift']:+.3f}"
            if faults and faults.get("incremental_lift") is not None
            else "-"
        )
        lines.append(
            f"{name:<17}{loeo['events_detected']:>8}/{loeo['events_total']:<2}"
            f"{loeo['mean_false_alarms_per_day']:>8.3f}"
            f"{loeo['mean_operating_false_alarms_per_day']:>8.3f}"
            f"{(f'{delay:.0f}' if delay is not None else '-'):>9}"
            f"{(f'{faults["detected"]}/{faults["cases"]}' if faults else '-'):>9}"
            f"{(faults['control_alarms'] if faults else '-'):>7}"
            f"{clean:>8}"
            f"{lift:>7}"
            f"{d['exceedance_check']['share_at_or_below_alpha']:>8.3f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="factorymind.anomaly.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="download, verify and validate MetroPT-3")
    train = sub.add_parser("train", help="run the experiment and save models")
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument(
        "--no-sensitivity", action="store_true", help="skip the exclusion-sensitivity re-run"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.command == "prepare":
        manifest = metropt3.prepare(settings.raw_dir, settings.processed_dir)
        print(json.dumps(manifest["raw_quality"], indent=2))
        return 0

    config = experiment.ExperimentConfig.load(args.config)
    telemetry = metropt3.load_processed(settings.processed_dir)
    output = experiment.run(config, telemetry)
    extra = None
    if not args.no_sensitivity:
        extra = {"sensitivity": experiment.run_sensitivity(config, telemetry)}
    run_dir = experiment.save(
        output, settings.artifacts_dir, settings.anomaly_model_name, extra=extra
    )
    print(_summary(output.results))
    if extra:
        print("sensitivity (no extreme-duty exclusion):", json.dumps(extra["sensitivity"]))
    print(f"results: {run_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
