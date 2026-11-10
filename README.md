# FactoryMind CDT

**An Uncertainty-Aware Cognitive Digital Twin for Evidence-Grounded Manufacturing Diagnosis,
Simulation, and Decision Support.**

SENSE → VALIDATE → UNDERSTAND → RETRIEVE → DIAGNOSE → ASSESS UNCERTAINTY → SIMULATE →
ESTIMATE IMPACT → HUMAN REVIEW → VERIFY → LEARN

> **Status: research prototype.** Implemented so far: the engineering foundation (M0) and a
> real-data compressor anomaly-detection pipeline with evaluation, inference API and React monitor
> (M1, see [docs/m1-anomaly-detection.md](docs/m1-anomaly-detection.md)). Evidence/RCA,
> uncertainty, simulation and the full decision workflow are later milestones. Nothing here has been
> deployed in a real factory; reported results come only from the evaluation pipeline in this
> repository.

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

### M1 results at a glance

On held-out MetroPT-3 test data (82 days, three operator air-leak reports), the selected Isolation
Forest detected 2/3 reports with no false alarms; PCA SPE detected 3/3 at 0.085 false alarms/day;
the plant's own low-pressure switch caught 1/3. Selection, protocol and limitations are documented
in [docs/m1-anomaly-detection.md](docs/m1-anomaly-detection.md).

## Repository layout

```
backend/    Python package `factorymind` (FastAPI service, data, models, evaluation)
frontend/   React + TypeScript application
docs/       Project documentation (reference material in docs/reference/)
```

Start with [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md) (current state and next task), then
[docs/architecture-review-post-m1.md](docs/architecture-review-post-m1.md) (review, roadmap,
acceptance criteria) and [docs/decisions.md](docs/decisions.md) (architecture decisions).

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
