# FactoryMind CDT

**An Uncertainty-Aware Cognitive Digital Twin for Evidence-Grounded Manufacturing Diagnosis,
Simulation, and Decision Support.**

SENSE → VALIDATE → UNDERSTAND → RETRIEVE → DIAGNOSE → ASSESS UNCERTAINTY → SIMULATE →
ESTIMATE IMPACT → HUMAN REVIEW → VERIFY → LEARN

> **Status: research prototype.** Implemented work packages (M0–M5) are tracked in
> [docs/roadmap-execution-tracker.md](docs/roadmap-execution-tracker.md), including real-data
> MetroPT-3 anomaly detection ([docs/m1-anomaly-detection.md](docs/m1-anomaly-detection.md),
> [docs/m2-wp0-detection-hardening.md](docs/m2-wp0-detection-hardening.md),
> [docs/m2-wp2-operating-regime-warning-separation.md](docs/m2-wp2-operating-regime-warning-separation.md)),
> OIDC/RBAC/audit persistence ([docs/m2-wp1-auth-persistence.md](docs/m2-wp1-auth-persistence.md)),
> evidence/RCA/simulation/copilot modules, and OT/edge preflight descriptors
> ([docs/m5-wp0-ot-edge-hardening.md](docs/m5-wp0-ot-edge-hardening.md)).
>
> **Important scope boundaries:** Nothing here has been deployed in a real factory, and
> **OPC-UA, MQTT, and historian connector I/O is not implemented** (`connector_io_status = "not_implemented"`):
> no industrial client libraries or live telemetry connections exist, endpoint checks inspect URL text
> only without DNS resolution, and the runtime loader still loads `model.joblib` (pickle) with
> adjacent `model_card.json` SHA-256 comparison rather than a verified non-pickle format.
> Reported metrics come solely from the evaluation pipelines and automated tests in this repository.

## Safety boundary

FactoryMind is **advisory and read-only** toward industrial systems. It never writes to PLCs,
changes setpoints or actuates equipment. Every recommendation requires human review. The API's
route set is checked by an automated test (`backend/tests/test_safety_boundary.py`).

## First use case

Early detection and diagnosis of **air leaks in a compressor Air Production Unit**, using the
public [MetroPT-3 dataset](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset)
(real operational telemetry, CC BY 4.0). The synthetic
[AI4I 2020 dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
(CC BY 4.0) is used as a controlled benchmark for diagnosis and uncertainty, because its failure
modes have known ground truth. Results on public, synthetic and simulated data are always reported
separately.

### Detection results at a glance

On real MetroPT-3 data (four operator air-leak reports), a physics rule on compressor load duty
detects 4/4 reports at 0.034 false alarms/day (0.044 operating-day FA/day) under WP0 nested
leave-one-event-out evaluation, and 4/4 at 0.026 FA/day (0.035 operating-day FA/day) with WP2
telemetry-integrity gating; `pca_spe` and signal integrity checks run alongside as separated
`sensor_consistency_warning` signals without inflating leak metrics. The plant's own low-pressure
switch caught 1/4. Protocol, results and limitations:
[docs/m2-wp0-detection-hardening.md](docs/m2-wp0-detection-hardening.md) and
[docs/m2-wp2-operating-regime-warning-separation.md](docs/m2-wp2-operating-regime-warning-separation.md)
(M1 record: [docs/m1-anomaly-detection.md](docs/m1-anomaly-detection.md)).

## Repository layout

```
backend/    Python package `factorymind` (FastAPI service, data, models, evaluation)
frontend/   React + TypeScript application
docs/       Project documentation (reference material in docs/reference/)
```

Large data, model caches and experiment runs are kept outside Git (see `.env.example`).

## Quick start

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/) 0.13+, Node.js 24.

```bash
cd backend
uv sync
uv run pytest
uv run python -m factorymind.anomaly.cli prepare   # download + validate MetroPT-3 (218 MB)
uv run python -m factorymind.anomaly.cli train     # train/evaluate detectors, save models
uv run uvicorn factorymind.api.app:app --reload
```

```bash
cd frontend
npm ci
npm run dev    # http://localhost:5173/anomaly
```

## Team

FactoryMind CDT is a graduation project. Team Leader: Mohamed Adel Ramadan. Technical direction,
architecture and implementation: Mohamed Osama.
