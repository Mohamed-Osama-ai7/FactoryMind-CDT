# M1 — Compressor anomaly detection (MetroPT-3)

> **Superseded protocol.** The evaluation protocol and results below are the M1 record. WP0 replaced
> the single-validation-event protocol with nested leave-one-event-out evaluation and added a physics
> baseline; see [m2-wp0-detection-hardening.md](m2-wp0-detection-hardening.md) for the current
> protocol, results and default detector.

First end-to-end pipeline: real dataset → validation → preprocessing → baselines → evaluation →
inference → FastAPI → React. Advisory only: outputs are anomaly scores and evidence for human
review; nothing controls equipment.

## Dataset

| | |
|---|---|
| Source | [UCI ML Repository #791, MetroPT-3](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset), DOI 10.24432/C5VW3R |
| License | CC BY 4.0 (attribution: Davari, Veloso, Ribeiro, Pereira, Gama) |
| Content | Real operational telemetry from the Air Production Unit (APU) compressor of a Porto metro train, 2020-02-01 to 2020-09-01 |
| Size | 1,516,948 rows, 15 signals (7 analogue: pressures, oil temperature, motor current; 8 digital valve/switch signals) |
| Sampling | ~0.1 Hz. The CSV is a 10× downsample of a 1 Hz log (its index column steps by 10), which reconciles the UCI page with the bundled PDF (15,169,480 points at 1 Hz) |
| Quality (measured) | No nulls, no duplicate timestamps, monotonic; 331 gaps > 60 s totalling 909.5 h (train out of service) |
| Labels | None. Four operator air-leak reports (F1 2020-04-18, F2 2020-05-29/30, F3 2020-06-05/07, F4 2020-07-15) are used as weak event labels for evaluation only |

Known source issues: the report table numbers two rows "#1" and its maintenance-date column is
ambiguous, so only report start/end times are used; the timezone of report times is not stated.

**Why MetroPT-3:** real (not synthetic) industrial condition-monitoring data under a permissive
license, with documented failure reports and a physically interpretable failure mode. During the
reported leaks the compressor's load duty is 0.52–0.84 versus a daily median of 0.105, while the
plant's own low-pressure switch stays silent for F1 and F2 — a realistic case where better
monitoring adds value.

## Reproduce

Requirements: see the repository README. Data and models are written outside Git (defaults
`D:/FactoryMind/data` and `D:/FactoryMind/artifacts`; override with `FACTORYMIND_DATA_DIR` /
`FACTORYMIND_ARTIFACTS_DIR`).

```bash
cd backend
uv run python -m factorymind.anomaly.cli prepare   # download (218 MB), SHA-256 verify, validate, Parquet
uv run python -m factorymind.anomaly.cli train     # experiment + models; prints the comparison table
```

`prepare` fails if the archive checksum, row count, duplicate timestamps or sampling period differ
from the documented contract. `train` is deterministic for a given config (seed 42); re-running it
reproduced identical metrics.

Experiment configuration: `backend/configs/metropt3_anomaly.json`. Outputs:
`artifacts/experiments/metropt3-anomaly/<version>/` (results.json, per-window scores) and
`artifacts/models/metropt3-anomaly/<version>-<detector>/` (model bundle + model card with SHA-256);
`CURRENT` names the served model.

## Method

**Windows and features** (`factorymind/anomaly/features.py`): clock-aligned 10-minute windows. A
window is valid only with ≥ 80 % of nominal samples and no internal gap > 60 s; gaps are never
interpolated. 46 features: mean/std/min/max of the 7 analogue signals, active fraction of the 8
digital signals, and four physics-motivated features — load duty (motor current > 5.5 A), load
starts, median TP3 slope while unloaded, and TP3–reservoir gap. Training and the API share this
code.

**Detectors** (`factorymind/anomaly/detectors.py`), all fitted on training windows only:

| Detector | Why it is included |
|---|---|
| `lps_rule` | The plant's existing low-pressure switch: the alarm the models must beat |
| `robust_z` | Transparent statistical baseline (max robust z-score) |
| `isolation_forest` | Standard unsupervised tabular anomaly detector |
| `pca_spe` | Classic process-monitoring method (PCA squared prediction error) |

**Protocol** (chronological, leakage-safe):

| Split | Period | Valid windows | Reports | Role |
|---|---|---|---|---|
| train | 2020-02-01 – 03-17 | 5,400 | — | fit scalers, feature profile, detectors |
| calibration | 03-17 – 04-01 | 1,912 | — | threshold = score quantile (false-alarm budget), never from labels |
| validation | 04-01 – 05-25 | 5,936 | F1 | choose quantile, persistence, variant and detector |
| test | 05-25 – 09-02 | 11,808 | F2, F3, F4 | evaluated once with frozen choices |

An alarm is raised after *k* consecutive windows above threshold (persistence). A report counts as
detected if an alarm is raised from 6 h before its start until its end. Alarm episodes that do not
touch any report (± 6 h before / 24 h after) count as false alarms, normalised per day of valid
data. Window-level PR-AUC/precision/recall are reported as secondary because window labels inherit
the reports' coarseness.

**Selection rule (declared before the test evaluation):** on validation only, maximise events
detected, then minimise false alarms/day, then maximise PR-AUC.

## Results

Error analysis after the first run found a defect, not a model weakness: several signals are
bimodal (e.g. TP2 ≈ 0 bar unloaded, ≈ 10 bar loaded), so median/MAD scaling made 49 % of
calibration windows saturate the robust z-score. The scale is now floored by a sigma estimate from
the training data's central 98 % span. One targeted tuning step then compared PCA variance
(0.90/0.95/0.99) and Isolation Forest `max_samples` (auto/2048) on validation. Startup exclusion was
considered and rejected: alarm episodes mostly start hours or days after a restart.

Final comparison (thresholds from calibration; choices from validation; test evaluated once):

| Detector | Split | Reports detected | Detection delay vs report start | False alarms/day | Window PR-AUC | Window P / R |
|---|---|---|---|---|---|---|
| lps_rule | validation | 0/1 | F1 missed | 0.000 | 0.02 | – / 0.00 |
| | test | 1/3 | F4 +240 min | 0.000 | 0.04 | 1.00 / 0.02 |
| robust_z | validation | 1/1 | F1 +90 min | 0.097 | 0.13 | 0.23 / 1.00 |
| | test | 3/3 | F2 +50, F3 +50, F4 −180 min | 0.341 | 0.05 | 0.10 / 0.94 |
| **isolation_forest** (selected) | validation | 1/1 | F1 +770 min | 0.073 | 0.40 | 0.12 / 0.04 |
| | test | 2/3 | F2 +120, F3 +50 min, F4 missed | 0.000 | 0.92 | 1.00 / 0.73 |
| pca_spe (99 % variance) | validation | 1/1 | F1 +90 min | 0.097 | 0.25 | 0.23 / 1.00 |
| | test | 3/3 | F2 +40, F3 +50, F4 0 min | 0.085 | 0.90 | 0.30 / 0.99 |

All alarm policies use persistence 6 (one hour of consecutive anomalous windows).

**Reading the results honestly**

- Both learned models clearly beat the plant's low-pressure switch, which caught one of four leaks.
- The selected Isolation Forest raised no false alarms in 82 test days and detected F2 and F3, but
  missed F4 (the shortest report, 4.5 h). Its validation detection of F1 came 12.8 h after the
  report start, which for F1 (reported as a whole day) may reflect the report's coarseness.
- PCA SPE detected all three test reports with shorter delays at about one false-alarm episode per
  12 days. **On test it looks better**, but it lost the pre-declared validation selection by one
  false-alarm episode, with only one validation event. Switching to it now would be tuning on the
  test set, so the selection stands and this is recorded as a limitation. A future milestone should
  pre-register a delay-aware rule and re-evaluate on unseen data.
- "False alarms" include high-load days with no report (e.g. 2020-05-19/20 before F2, 2020-06-22);
  these may be unreported anomalies rather than model errors.
- The Isolation Forest score sits close to its threshold during sustained anomalies (≈ 0.73 vs
  0.726), so the decision margin is small.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/anomaly/model` | Model card summary: version, dataset, license, alarm policy, validation/test metrics, limitations, replay range, operator reports |
| POST | `/api/v1/anomaly/score` | Score one block of samples (MetroPT-3 schema, ≤ 2000 samples, naive strictly increasing timestamps) as one window; returns status `normal` / `anomalous` / `insufficient_data`, score, threshold and top-5 feature deviations |
| GET | `/api/v1/anomaly/replay?start=&end=` | Score recorded telemetry (≤ 7 days) window by window with alarm persistence; returns windows, alarm episodes and overlapping operator reports |

Invalid input (unknown/missing fields, non-binary digital values, physically implausible values,
non-chronological or timezone-aware timestamps) returns 422. A missing, incomplete or
checksum-mismatched model returns 503; replay without processed data returns 503. `POST /score` is
stateless and is the only allow-listed non-GET route (`tests/test_safety_boundary.py`).

Feature evidence is the deviation from the training distribution (robust z) — useful for review,
not a root cause.

## Frontend

`/anomaly` shows the model card, a replay form (start, duration, presets around each operator
report), an ECharts score timeline with threshold, alarm episodes and reports, and an alarm-episode
table with inline evidence. All data comes from the live API.

## Limitations

- One compressor, one failure type (air leak), four coarse reports; results do not establish
  performance on other assets or failure modes.
- Validation contains a single event, so model selection is fragile (see Results).
- Unreported anomalies are counted as false alarms; report timezone is unknown.
- `POST /score` treats each request as one window; alarm persistence and restart context apply only
  in replay.
- Replay computes alarm persistence within the requested range only.
