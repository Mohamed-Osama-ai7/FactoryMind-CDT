# FactoryMind CDT

**An Uncertainty-Aware Cognitive Digital Twin for Evidence-Grounded Manufacturing Diagnosis,
Simulation, and Decision Support.**

SENSE → VALIDATE → UNDERSTAND → RETRIEVE → DIAGNOSE → ASSESS UNCERTAINTY → SIMULATE →
ESTIMATE IMPACT → HUMAN REVIEW → VERIFY → LEARN

> **Status: early implementation (research prototype).** The repository currently contains the
> engineering foundation (backend service, React application, CI). Models, evidence/RCA,
> uncertainty, simulation and the end-to-end workflow are being built milestone by milestone.
> Nothing here has been deployed in a real factory, and no performance result is claimed until it
> is produced by the evaluation pipeline in this repository.

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
uv run uvicorn factorymind.api.app:app --reload
```

```bash
cd frontend
npm ci
npm run dev
```

## Team

FactoryMind CDT is a graduation project. Team Leader: Mohamed Adel Ramadan. Technical direction,
architecture and implementation: Mohamed Osama.
