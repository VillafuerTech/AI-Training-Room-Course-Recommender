# IBM ML Capstone — Recommendation System

## Overview
Production-ready scaffold for a recommender system (implicit/explicit feedback) aligned with CRISP-DM.

## Structure
- `data/`: raw/interim/processed/external
- `notebooks/`: EDA & experiments
- `src/recsys/`: package code
- `models/`: persisted models
- `reports/`: figures and final artifacts
- `configs/`: YAML/JSON configs
- `tests/`: unit tests
- `scripts/`: CLI utilities

## Quickstart
```bash
# create venv (optional)
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# format & lint
make fmt lint

# run pipeline (example)
python -m src.recsys.pipelines.run_all --config configs/default.yaml
