#!/usr/bin/env python3
"""
Create data/external/courses.csv (columns: item,title)

Sources (in priority order):
1) Official IBM catalog (course_processed.csv) via HTTPS
2) Hand-maintained overrides for common IDs (top/popular)
3) Fallback: readable title from the code (underscores/hyphens → spaces, title case)

Outputs:
- data/external/courses.csv
- logs/missing_courses.txt (any items we couldn't title from the official source)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# --- Paths
ROOT = Path(__file__).resolve().parents[1]
RATINGS_PATH = ROOT / "data" / "external" / "course_ratings.csv"
OUT_PATH = ROOT / "data" / "external" / "courses.csv"
LOGS_DIR = ROOT / "logs"
MISS_PATH = LOGS_DIR / "missing_courses.txt"

# --- Official catalog (same family as the course labs)
CATALOG_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-ML321EN-SkillsNetwork/labs/datasets/course_processed.csv"
)

# --- Optional overrides for frequent/popular IDs (exactly as you want them displayed)
OVERRIDES = {
    "PY0101EN": "Python for Data Science",
    "DS0101EN": "Introduction to Data Science",
    "BD0101EN": "Big Data 101",
    "BD0111EN": "Hadoop 101",
    "DA0101EN": "Data Analysis with Python",
    "DS0103EN": "Data Science Methodology",
    "ML0101ENv3": "Machine Learning with Python",
    "BD0211EN": "Spark Fundamentals I",
    "DS0105EN": "Data Science Hands-On with Open Source Tools",
    "BC0101EN": "Blockchain Essentials",
    "DV0101EN": "Data Visualization with Python",
    "ML0115EN": "Deep Learning 101",
    "CB0103EN": "Build Your Own Chatbot",
    "RP0101EN": "R for Data Science",
    "ST0101EN": "Statistics 101",
    "CC0101EN": "Introduction to Cloud",
    "CO0101EN": "Docker Essentials: A Developer Introduction",
    "DB0101EN": "SQL and Relational Databases 101",
    "BD0115EN": "MapReduce and YARN",
    "DS0301EN": "Data Privacy Fundamentals",
}


def prettify(code: str) -> str:
    """Readable title from an item code as a last-resort fallback."""
    # keep common suffixes (EN, v3) but insert spaces where appropriate
    t = code.replace("_", " ").replace("-", " ")
    # Simple heuristics: split letters+digits boundaries
    import re

    t = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", t)
    t = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", t)
    # Title-case but preserve common acronyms
    title = t.title()
    for acro in ["AI", "ML", "SQL", "GPU", "NLP", "R", "IoT"]:
        title = title.replace(acro.title(), acro)
    return title.strip()


def main() -> int:
    if not RATINGS_PATH.exists():
        print(
            f"ERROR: {RATINGS_PATH} not found. Export ratings first.", file=sys.stderr
        )
        return 2

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1) Course IDs actually present in your project
    ratings = pd.read_csv(RATINGS_PATH, usecols=["user", "item", "rating"])
    items = pd.DataFrame({"item": sorted(ratings["item"].astype(str).unique())})

    # 2) Try official catalog (COURSE_ID, TITLE)
    catalog = None
    try:
        cat_raw = pd.read_csv(CATALOG_URL, dtype=str, usecols=["COURSE_ID", "TITLE"])
        catalog = (
            cat_raw.drop_duplicates(subset=["COURSE_ID"])
            .rename(columns={"COURSE_ID": "item", "TITLE": "title"})
            .assign(item=lambda d: d["item"].astype(str))
        )
    except Exception as e:
        print(
            f"WARNING: could not download official catalog ({e}). Proceeding with overrides/fallbacks.",
            file=sys.stderr,
        )

    # Start assembling mapping
    out = items.copy()
    out["title"] = None

    # Join from official catalog if available
    if catalog is not None:
        out = out.merge(catalog, on="item", how="left", suffixes=("", "_cat"))
        out["title"] = out["title"].where(out["title"].notna(), out.get("title_cat"))
        if "title_cat" in out:
            out = out.drop(columns=["title_cat"])

    # Apply overrides
    mask_missing = out["title"].isna()
    if mask_missing.any():
        out.loc[mask_missing, "title"] = [
            OVERRIDES.get(i, None) for i in out.loc[mask_missing, "item"]
        ]

    # Fallback prettify for any remaining gaps
    mask_missing = out["title"].isna()
    if mask_missing.any():
        out.loc[mask_missing, "title"] = [
            prettify(i) for i in out.loc[mask_missing, "item"]
        ]

    # Save and log any that still look suspicious (rare)
    out = out[["item", "title"]].drop_duplicates()
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out):,} rows → {OUT_PATH}")

    # Log entries that were not from the official catalog (for your review)
    if catalog is not None:
        from_cat = set(catalog["item"])
        guessed = out[~out["item"].isin(from_cat)]
        if not guessed.empty:
            MISS_PATH.write_text(
                "\n".join(
                    [f"{r.item},{r.title}" for r in guessed.itertuples(index=False)]
                )
            )
            print(f"Logged {len(guessed)} non-catalog titles → {MISS_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
