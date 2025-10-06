# Personalized Course Recommender (ML321EN Capstone)

A production-quality, local-first course recommender that combines **content signals** and **collaborative signals** (embeddings) with an optional **hybrid** ranker. This repository includes an interactive Streamlit application, experiment notebooks, trained models, and a Beamer presentation.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Data Files & Schemas](#data-files--schemas)
- [Running the App](#running-the-app)
- [Algorithms & Implementation](#algorithms--implementation)
- [Evaluation](#evaluation)
- [Repository Layout](#repository-layout)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Features

- **Streamlit application** (`streamlit_app.py`)

  - Recommenders: **Popularity**, **Embedding Affinity** (user–item embeddings), **Item–Item** (cosine), and a **Hybrid** blend
  - Existing-user and **cold-start** flows
  - CSV export of recommendations
  - Quick **Leave-One-Out** (LOO) evaluation
  - **PCA** visualization of similar items (for intuition only)

- **Exact course titles** via `data/external/courses.csv` (no auto-mangling when metadata is provided)

- **Local upload** support with remote fallbacks to public IBM ML321EN datasets

- **Experiments & models**

  - Baseline embeddings in `data/processed/*embeddings_baseline.csv`
  - Trained models in `models/` and experiment logs in `logs/`

- **Presentation**

  - Beamer slides: `docs/Presentation/presentation.pdf`

---

## Quick Start

### 1) Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Data (local defaults)

This repo includes baseline CSVs under `data/processed/` and a ratings file at `data/external/course_ratings.csv`.

To ensure **correct, human-readable course titles** in the UI and exports, provide:

```
data/external/courses.csv     # columns: item,title
```

Build it from public metadata with the helper script:

```bash
python scripts/make_courses_csv.py --out data/external/courses.csv
```

> When `courses.csv` is present, the app uses titles **verbatim**.
> If it’s missing, the app falls back to a safe ID-prettifying heuristic.

---

## Data Files & Schemas

You can use the bundled files or upload your own in the Streamlit sidebar.

| File                                          | Required | Columns / Notes                                                                                     |
| --------------------------------------------- | :------: | --------------------------------------------------------------------------------------------------- |
| `data/external/course_ratings.csv`            |  Yes\*   | `user,item,rating`. Lab ratings are integers (3/4/5); the app accepts any numeric scale.            |
| `data/processed/user_embeddings_baseline.csv` |  Yes\*   | `user` **or** `user_id` + N numeric feature columns. Feature names are normalized internally.       |
| `data/processed/item_embeddings_baseline.csv` |  Yes\*   | `item` **or** `item_id` + N numeric feature columns. Embedding dimension is inferred automatically. |
| `data/external/courses.csv`                   |  Strong  | `item,title`. Guarantees exact titles in the UI and exported CSVs.                                  |

- You may also upload these CSVs directly in the Streamlit sidebar to override defaults.

**Example `courses.csv`**

```csv
item,title
PY0101EN,Python for Data Science
DS0101EN,Introduction to Data Science
ML0101ENv3,Machine Learning with Python
...
```

---

## Running the App

```bash
streamlit run streamlit_app.py
```

Open the URL printed by Streamlit in your terminal.

**Usage overview**

- **Data Source (sidebar)**: upload `ratings.csv`, `user_embeddings.csv`, `course_embeddings.csv`, and optional `courses.csv`.
  If not uploaded, the app loads local files first, then public fallbacks.

- **Recommendation Settings**:

  - User mode: Existing user (selected from ratings) or New user (select a few liked courses)
  - Recommender: Popularity / Embedding Affinity / Item–Item (cosine) / Hybrid
  - Hybrid: tune weights; they are normalized at runtime
  - Top-N: number of recommendations
  - Advanced: toggle “exclude seen items” for existing users

- **Recommendations**: results table with `item`, `title`, overall `score`, and component scores (where applicable); keyword filter; CSV download.

- **Evaluation**: quick LOO evaluation with Hit-Rate@K, Precision@K, Recall@K.

- **Visualization**: PCA projection of recommended item embeddings (exploratory).

---

## Algorithms & Implementation

- **Popularity**

  - Score = α·`enrolls` + (1−α)·`mean_rating` (default α=0.7)
  - Useful baseline and cold-start prior.

- **Embedding Affinity**

  - If a user vector exists: cosine similarity between user embedding and item embeddings.
  - For cold-start: synthesize a user vector by averaging liked items’ embeddings.

- **Item–Item (cosine)**

  - Precompute item–item cosine similarities over item embeddings.
  - Aggregate neighbor scores from the user’s liked/seen items (excludes the item itself and previously seen items).

- **Hybrid**

  - Weighted blend of normalized component scores:
    `score = w_pop * pop + w_aff * aff + w_i2i * i2i` (weights sum to 1).

Implementation details (selected):

- Defensive neighbor selection on small catalogs (bounds top-k by available items)
- Strict preservation of titles when `courses.csv` is provided
- Lightweight caching for CSV loads in Streamlit

---

## Evaluation

A lightweight **Leave-One-Out** (LOO) routine (sampled users for speed):

- Split per user: hold out one positive interaction; recommend with remaining history
- Report **Hit-Rate@K**, **Precision@K**, **Recall@K**
- Works for Popularity, Embedding Affinity, Item–Item, and Hybrid

For comprehensive benchmarking, consider additional ranking metrics (MAP@n, nDCG@n) and multiple random seeds.

---

## Repository Layout

```
configs/
data/
  external/            # ratings, courses.csv (exact titles)
  interim/
  processed/           # embeddings, metrics, tables
docs/
  Presentation/        # Beamer slides (.pdf, .tex)
logs/                  # experiment metadata and logs
models/                # saved .keras / .joblib models
notebooks/             # labs and experiments
reports/
  figures/             # figures for notebooks/presentation
scripts/
  make_courses_csv.py
src/
  recsys/              # package skeleton (data/features/models/pipelines/utils)
streamlit_app.py       # Streamlit UI entry point
tests/
tmp/
```

---

## Reproducibility

- Processed experiment tables live in `data/processed/` (e.g., `model_comparison_top10.csv`, `recommender_eval.csv`).
- Figures for the report are under `reports/figures/` and referenced by the Beamer deck.
- Notebooks pin a random seed where appropriate; see notebook headers for specifics.

---

## Contributing

1. Create a feature branch from `main`.
2. Follow the project’s code style and add tests where practical.
3. Open a pull request with a concise description and rationale.

Please avoid committing large datasets to the repository. Place data under `data/` and rely on `.gitignore` rules already provided.

---

## License

This project is released under the **MIT License** Copyright © 2025
**Roberto Villafuerte Carrillo**

See the [`LICENSE`](./LICENSE) file for full terms.

---
