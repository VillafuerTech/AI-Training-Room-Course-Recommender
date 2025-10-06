# streamlit_app.py
# ---------------------------------------------------------
# Personalized Course Recommender — ML321EN Capstone
# Strategies: Popularity, Embedding Affinity, Item–Item (cosine), Hybrid
# Data: local-first with optional upload and remote fallback for lab defaults
# ---------------------------------------------------------

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Page config and light styling
# -------------------------------
st.set_page_config(page_title="Course Recommender — ML Capstone", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
      h1, h2, h3 {font-weight: 700;}
      .stCaption {color: #5a5a5a;}
      .metric-small {font-size: 0.9rem; color: #666;}
      .tight-table thead th {font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)

RS = 123
np.random.seed(RS)

# -------------------------------
# Default sources (lab fallbacks)
# -------------------------------
RATINGS_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBMSkillsNetwork-ML0321EN-Coursera/labs/v2/module_3/ratings.csv"
)
USER_EMB_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-ML321EN-SkillsNetwork/labs/datasets/user_embeddings.csv"
)
ITEM_EMB_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-ML321EN-SkillsNetwork/labs/datasets/course_embeddings.csv"
)

# Local default metadata (item -> title)
META_LOCAL_DEFAULT = Path("data/external/courses.csv")


# -------------------------------
# Utilities: normalize columns
# -------------------------------
def normalize_user_emb(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    id_col = (
        "user"
        if "user" in df.columns
        else ("user_id" if "user_id" in df.columns else None)
    )
    if id_col is None:
        raise ValueError("User embedding file must include 'user' or 'user_id' column.")
    feat_cols = [c for c in df.columns if c != id_col]
    rename = {id_col: "user"}
    rename.update({c: f"UFeature{i}" for i, c in enumerate(feat_cols)})
    out = df.rename(columns=rename)
    return out, [f"UFeature{i}" for i in range(len(feat_cols))]


def normalize_item_emb(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    id_col = (
        "item"
        if "item" in df.columns
        else ("item_id" if "item_id" in df.columns else None)
    )
    if id_col is None:
        raise ValueError("Item embedding file must include 'item' or 'item_id' column.")
    feat_cols = [c for c in df.columns if c != id_col]
    rename = {id_col: "item"}
    rename.update({c: f"CFeature{i}" for i, c in enumerate(feat_cols)})
    out = df.rename(columns=rename)
    return out, [f"CFeature{i}" for i in range(len(feat_cols))]


MINOR_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "for",
    "nor",
    "on",
    "at",
    "to",
    "from",
    "by",
    "of",
    "with",
    "in",
    "into",
    "over",
    "vs",
    "via",
}
ACRONYM_FIXES = {
    "SQL": "SQL",
    "IBM": "IBM",
    "AI": "AI",
    "ML": "ML",
    "NLP": "NLP",
    "GPU": "GPU",
    "CPU": "CPU",
    "IoT": "IoT",
    "DB2": "DB2",
    "AWS": "AWS",
    "GCP": "GCP",
    "API": "API",
    "PCA": "PCA",
    "KNN": "KNN",
    "SVM": "SVM",
    "XGBoost": "XGBoost",
    "NoSQL": "NoSQL",
    "HTML": "HTML",
    "CSS": "CSS",
    "JSON": "JSON",
    "PyTorch": "PyTorch",
    "TensorFlow": "TensorFlow",
    "Keras": "Keras",
    "NumPy": "NumPy",
    "Pandas": "Pandas",
    "JavaScript": "JavaScript",
    "Node.js": "Node.js",
    "C++": "C++",
    "R": "R",
    "Hadoop": "Hadoop",
    "Spark": "Spark",
}

_ROMAN_RE = re.compile(r"\b(i|ii|iii|iv|v|vi|vii|viii|ix|x)\b", flags=re.IGNORECASE)


def _smart_title(text: str) -> str:
    """Title-case with proper handling of acronyms, roman numerals, hyphens, and minor words."""
    t = re.sub(r"\s+", " ", str(text).strip())
    if not t:
        return t

    words = t.split(" ")
    out = []
    for i, w in enumerate(words):
        # hyphenated words: title-case each piece
        parts = w.split("-")
        norm_parts = []
        for p in parts:
            if not p:
                norm_parts.append(p)
                continue
            lower = p.lower()
            if i not in (0, len(words) - 1) and lower in MINOR_WORDS:
                norm_parts.append(lower)
            else:
                norm_parts.append(p[:1].upper() + p[1:].lower())
        ww = "-".join(norm_parts)
        out.append(ww)
    titled = " ".join(out)

    # Roman numerals
    titled = _ROMAN_RE.sub(lambda m: m.group(0).upper(), titled)

    # Common technology names/acronyms
    def _apply_acronyms(txt: str) -> str:
        for canonical in ACRONYM_FIXES:
            pattern = re.compile(rf"\b{re.escape(canonical)}\b", re.IGNORECASE)
            txt = pattern.sub(ACRONYM_FIXES[canonical], txt)
        # special cases that often appear after generic title-casing
        txt = txt.replace("Javascript", "JavaScript")
        txt = txt.replace("Node.Js", "Node.js")
        txt = txt.replace("Pytorch", "PyTorch")
        txt = txt.replace("Tensorflow", "TensorFlow")
        txt = txt.replace("Iot", "IoT")
        txt = txt.replace("Nosql", "NoSQL")
        txt = txt.replace("Db2", "DB2")
        txt = txt.replace("Ibm", "IBM")
        return txt

    return _apply_acronyms(titled)


def _prettify_from_code(item_code: str) -> str:
    """Readable title from an item code (fallback when no metadata)."""
    x = item_code.replace("_", " ").replace("-", " ")
    x = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", x)  # A123 -> A 123
    x = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", x)  # 123A -> 123 A
    return _smart_title(x)


# -------------------------------
# Data loading
# -------------------------------
@st.cache_data(show_spinner=False)
def load_csv_local_or_url(local_path: Optional[Path], url: str) -> pd.DataFrame:
    try:
        if local_path and local_path.exists():
            return pd.read_csv(local_path)
        return pd.read_csv(url)
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV from {local_path or url}: {e}")


def build_title_map(
    items_df: pd.DataFrame, meta_df: Optional[pd.DataFrame]
) -> pd.Series:
    """
    Returns a Series indexed by item id with the exact course title.
    Never alters capitalization; if no metadata is available, fall back to item id itself.
    """
    if meta_df is not None and {"item", "title"}.issubset(meta_df.columns):
        return (
            meta_df[["item", "title"]]
            .drop_duplicates(subset=["item"])
            .set_index("item")["title"]
        )
    # Fallback: keep raw item ID as title (no prettifying)
    return pd.Series(items_df["item"].values, index=items_df["item"], name="title")


# -------------------------------
# Recommenders
# -------------------------------
def popularity_recommender(
    ratings: pd.DataFrame, exclude: List[str], top_n: int, min_enrolls: int = 0
) -> pd.DataFrame:
    pop = (
        ratings.groupby("item")
        .agg(enrolls=("rating", "count"), mean_rating=("rating", "mean"))
        .reset_index()
    )
    if min_enrolls > 0:
        pop = pop[pop["enrolls"] >= min_enrolls]
    pop["score"] = pop["enrolls"] * 0.7 + pop["mean_rating"] * 0.3
    if exclude:
        pop = pop[~pop["item"].isin(exclude)]
    return pop.sort_values("score", ascending=False).head(top_n)[
        ["item", "score", "enrolls", "mean_rating"]
    ]


def get_user_vector(
    user_id: int, users_df: pd.DataFrame, u_cols: List[str]
) -> Optional[np.ndarray]:
    row = users_df.loc[users_df["user"] == user_id]
    if len(row) == 0:
        return None
    return row[u_cols].values.astype("float32")[0]


def embedding_affinity_recommender(
    user_id: Optional[int],
    liked_items: List[str],
    users_df: pd.DataFrame,
    items_df: pd.DataFrame,
    u_cols: List[str],
    c_cols: List[str],
    seen: List[str],
    top_n: int,
    use_cosine: bool = True,
) -> pd.DataFrame:
    # Get user vector, or synthesize from liked items (cold start)
    u_vec = get_user_vector(user_id, users_df, u_cols) if user_id is not None else None
    if u_vec is None:
        if not liked_items:
            return pd.DataFrame(columns=["item", "score"])
        liked = items_df[items_df["item"].isin(liked_items)]
        if liked.empty:
            return pd.DataFrame(columns=["item", "score"])
        u_vec = liked[c_cols].mean(axis=0).values.astype("float32")

    C = items_df[c_cols].values.astype("float32")
    if use_cosine:
        scores = cosine_similarity(C, u_vec[None, :]).reshape(-1)
    else:
        scores = C @ u_vec

    out = items_df[["item"]].copy()
    out["score"] = scores
    if seen:
        out = out[~out["item"].isin(seen)]
    return out.sort_values("score", ascending=False).head(top_n)


def top_k_neighbors_from_row(
    row: np.ndarray, k: int, exclude_idx: Optional[int] = None
) -> np.ndarray:
    """
    Robust top-k selector for a similarity row.
    Excludes self index, uses argpartition on -row to avoid negative kth errors.
    """
    r = row.astype(float, copy=True)
    if exclude_idx is not None:
        r[exclude_idx] = -np.inf
    n = r.shape[0]
    k = max(0, min(k, n - (1 if exclude_idx is not None else 0)))
    if k == 0:
        return np.empty(0, dtype=int)
    kth = min(k, n - 1) - 1
    idx = np.argpartition(-r, kth)[:k]
    return idx[np.argsort(r[idx])[::-1]]


def item_item_recommender(
    liked_items: List[str],
    items_df: pd.DataFrame,
    c_cols: List[str],
    seen: List[str],
    top_n: int,
    top_sim_k: int = 200,
) -> pd.DataFrame:
    if not liked_items:
        return pd.DataFrame(columns=["item", "score"])

    idx = pd.Index(items_df["item"])
    C = items_df[c_cols].values.astype("float32")
    sim = cosine_similarity(C)

    liked_idx = [idx.get_loc(i) for i in liked_items if i in idx]
    if not liked_idx:
        return pd.DataFrame(columns=["item", "score"])

    agg = np.zeros(len(idx), dtype="float32")
    for li in liked_idx:
        neighbors = top_k_neighbors_from_row(sim[li], top_sim_k, exclude_idx=li)
        agg[neighbors] += sim[li, neighbors]

    mask_excl = set(seen) | set(liked_items)
    out = pd.DataFrame({"item": idx, "score": agg})
    out = out[~out["item"].isin(mask_excl)]
    return out.sort_values("score", ascending=False).head(top_n)


def hybrid_recommender(
    user_id: Optional[int],
    liked_items: List[str],
    ratings: pd.DataFrame,
    users_df: pd.DataFrame,
    items_df: pd.DataFrame,
    u_cols: List[str],
    c_cols: List[str],
    seen: List[str],
    top_n: int,
    w_pop: float = 0.15,
    w_aff: float = 0.55,
    w_i2i: float = 0.30,
    min_enrolls: int = 0,
) -> pd.DataFrame:
    # Component candidates (oversampled, then trimmed after blend)
    cand_n = max(top_n * 5, 200)

    pop = popularity_recommender(
        ratings, exclude=seen, top_n=cand_n, min_enrolls=min_enrolls
    )
    aff = embedding_affinity_recommender(
        user_id,
        liked_items,
        users_df,
        items_df,
        u_cols,
        c_cols,
        seen,
        top_n=cand_n,
        use_cosine=True,
    )
    i2i = item_item_recommender(liked_items, items_df, c_cols, seen, top_n=cand_n)

    def norm(df: pd.DataFrame, col="score") -> pd.DataFrame:
        if df.empty:
            return df.assign(score_norm=0.0)
        s = df[col].to_numpy()
        lo, hi = float(np.min(s)), float(np.max(s))
        if hi <= lo:
            return df.assign(score_norm=np.zeros_like(s, dtype="float32"))
        return df.assign(score_norm=((s - lo) / (hi - lo)).astype("float32"))

    pop = norm(pop).rename(columns={"score_norm": "pop"})
    aff = norm(aff).rename(columns={"score_norm": "aff"})
    i2i = norm(i2i).rename(columns={"score_norm": "i2i"})

    blended = (
        pop[["item", "pop"]]
        .merge(aff[["item", "aff"]], on="item", how="outer")
        .merge(i2i[["item", "i2i"]], on="item", how="outer")
        .fillna(0.0)
    )
    s = max(1e-12, (w_pop + w_aff + w_i2i))
    w_pop, w_aff, w_i2i = w_pop / s, w_aff / s, w_i2i / s

    blended["score"] = (
        w_pop * blended["pop"] + w_aff * blended["aff"] + w_i2i * blended["i2i"]
    )
    blended = blended[~blended["item"].isin(seen)]
    return blended.sort_values("score", ascending=False).head(top_n)[
        ["item", "score", "pop", "aff", "i2i"]
    ]


# -------------------------------
# Header
# -------------------------------
st.title("Personalized Course Recommender")
st.caption(
    "ML321EN Capstone — Content, Collaborative (Embeddings), and Hybrid strategies"
)

# -------------------------------
# Sidebar — Data sources
# -------------------------------
with st.sidebar:
    st.header("Data Source")

    st.write("Upload your own CSVs to override defaults.")
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        up_r = st.file_uploader(
            "ratings.csv", type=["csv"], help="Columns: user,item,rating"
        )
        up_ue = st.file_uploader("user_embeddings.csv", type=["csv"])
    with col_up2:
        up_meta = st.file_uploader(
            "courses.csv (optional)", type=["csv"], help="Columns: item,title"
        )
        up_ie = st.file_uploader("course_embeddings.csv", type=["csv"])

# Load data (uploaded overrides)
try:
    if up_r is not None:
        ratings = pd.read_csv(up_r)[["user", "item", "rating"]].copy()
    else:
        ratings = load_csv_local_or_url(
            Path("data/external/course_ratings.csv"), RATINGS_URL
        )[["user", "item", "rating"]].copy()

    if up_ue is not None:
        user_raw = pd.read_csv(up_ue)
    else:
        user_raw = load_csv_local_or_url(
            Path("data/processed/user_embeddings_baseline.csv"), USER_EMB_URL
        )

    if up_ie is not None:
        item_raw = pd.read_csv(up_ie)
    else:
        item_raw = load_csv_local_or_url(
            Path("data/processed/item_embeddings_baseline.csv"), ITEM_EMB_URL
        )

    users, u_cols = normalize_user_emb(user_raw)
    items, c_cols = normalize_item_emb(item_raw)

    if up_meta is not None:
        meta = pd.read_csv(up_meta)
        if not {"item", "title"}.issubset(meta.columns):
            st.warning(
                "Uploaded courses.csv is missing columns {item,title}. Ignoring."
            )
            meta = None
    else:
        meta = pd.read_csv(META_LOCAL_DEFAULT) if META_LOCAL_DEFAULT.exists() else None

    title_map = build_title_map(items, meta)

except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()

# Quick data summary
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Users", f"{ratings['user'].nunique():,}")
col_b.metric("Courses", f"{ratings['item'].nunique():,}")
col_c.metric("Interactions", f"{len(ratings):,}")
col_d.metric("Embedding Dim (items)", len(c_cols))

st.divider()

# -------------------------------
# Controls and output
# -------------------------------
left, right = st.columns([0.40, 0.60], gap="large")

with left:
    st.subheader("Recommendation Settings")

    user_ids = ratings["user"].unique()
    mode = st.radio(
        "User mode", ["Existing user", "New (cold-start) user"], horizontal=True
    )

    chosen_user: Optional[int] = None
    liked_items: List[str] = []
    exclude_seen = True

    if mode == "Existing user":
        chosen_user = int(st.selectbox("Select user", sorted(user_ids)))
        user_hist = ratings[ratings["user"] == chosen_user].sort_values(
            "rating", ascending=False
        )
        seen_items = user_hist["item"].tolist()
        exclude_seen = st.toggle("Exclude previously seen items", value=True)
        st.caption(f"Interactions for user {chosen_user}: {len(user_hist)}")
        if len(user_hist):
            show_hist = (
                user_hist.merge(
                    title_map.rename("title"),
                    left_on="item",
                    right_index=True,
                    how="left",
                )
                .assign(title=lambda d: d["title"].fillna(d["item"]))
                .loc[:, ["item", "title", "rating"]]
                .head(12)
            )
            st.dataframe(show_hist, use_container_width=True, height=260)
    else:
        chosen_user = None
        # Show up to 800 options for usability; display titles but return item ids
        all_items = items["item"].tolist()
        label_lookup = title_map.to_dict()

        def item_label(x: str) -> str:
            t = label_lookup.get(x, x)
            return t if t else x

        liked_items = st.multiselect(
            "Select a few courses you like (2–10)",
            options=all_items[:800],
            max_selections=15,
            format_func=item_label,
        )
        seen_items = liked_items
        exclude_seen = True

    top_n = st.slider("Top-N recommendations", 5, 50, 10, step=1)

    algo = st.selectbox(
        "Recommender",
        ["Hybrid", "Embedding Affinity", "Item–Item (cosine)", "Popularity"],
        index=0,
        help="Hybrid blends popularity, embedding affinity, and item–item similarity.",
    )

    if algo == "Hybrid":
        st.markdown("**Hybrid weights**")
        w_pop = st.slider("Popularity", 0.0, 1.0, 0.15, 0.05)
        w_aff = st.slider("Embedding Affinity", 0.0, 1.0, 0.55, 0.05)
        w_i2i = st.slider("Item–Item", 0.0, 1.0, 0.30, 0.05)
    else:
        w_pop = w_aff = w_i2i = None

    with st.expander("Advanced settings"):
        min_enrolls = st.number_input(
            "Popularity: minimum enroll count to consider",
            min_value=0,
            max_value=10000,
            value=0,
            step=50,
            help="Filter very rare courses when using popularity or hybrid.",
        )
        use_cos = st.checkbox("Use cosine for embedding affinity", value=True)
        n_items_total = items.shape[0]
        default_k = min(200, max(1, n_items_total - 1))
        top_sim_k = st.slider(
            "Item–Item neighbor pool (K)",
            10,
            max(50, min(1000, n_items_total - 1)),
            default_k,
            step=10,
            help="Larger K is more thorough but slower. Capped to available items.",
        )

    st.divider()
    with st.expander("Quick Offline Evaluation (Leave-One-Out)"):
        k_eval = st.slider("K for top-K", 5, 20, 10)
        n_eval = st.slider("Users sampled", 100, 2000, 500, step=100)
        if st.button("Run evaluation", use_container_width=True):
            with st.spinner("Evaluating..."):
                metrics = leave_one_out_eval_topk(
                    ratings,
                    items,
                    users,
                    list(u_cols),
                    list(c_cols),
                    recommender=algo,
                    k=k_eval,
                    users_sample=n_eval,
                )
            st.success(f"Used users: {metrics['users']}")
            st.write(
                f"Hit-Rate@{k_eval}: {metrics['hit_rate']:.3f}  |  "
                f"Precision@{k_eval}: {metrics['precision']:.3f}  |  "
                f"Recall@{k_eval}: {metrics['recall']:.3f}"
            )

with right:
    st.subheader("Recommendations")

    # Compute recommendations
    if algo == "Popularity":
        recs = popularity_recommender(
            ratings,
            exclude=seen_items if exclude_seen else [],
            top_n=top_n,
            min_enrolls=min_enrolls,
        )
    elif algo == "Embedding Affinity":
        recs = embedding_affinity_recommender(
            user_id=chosen_user,
            liked_items=liked_items,
            users_df=users,
            items_df=items,
            u_cols=list(u_cols),
            c_cols=list(c_cols),
            seen=seen_items if exclude_seen else [],
            top_n=top_n,
            use_cosine=use_cos,
        )
    elif algo == "Item–Item (cosine)":
        # For existing user, use their history; for cold start, use liked items
        base_likes = (
            ratings.loc[ratings["user"] == chosen_user, "item"].tolist()
            if chosen_user is not None
            else liked_items
        )
        recs = item_item_recommender(
            liked_items=base_likes,
            items_df=items,
            c_cols=list(c_cols),
            seen=seen_items if exclude_seen else [],
            top_n=top_n,
            top_sim_k=top_sim_k,
        )
    else:
        base_likes = (
            ratings.loc[ratings["user"] == chosen_user, "item"].tolist()
            if chosen_user is not None
            else liked_items
        )
        recs = hybrid_recommender(
            user_id=chosen_user,
            liked_items=base_likes,
            ratings=ratings,
            users_df=users,
            items_df=items,
            u_cols=list(u_cols),
            c_cols=list(c_cols),
            seen=seen_items if exclude_seen else [],
            top_n=top_n,
            w_pop=w_pop,
            w_aff=w_aff,
            w_i2i=w_i2i,
            min_enrolls=min_enrolls,
        )

    # Attach exact titles
    out = recs.merge(
        title_map.rename("title"), left_on="item", right_index=True, how="left"
    )
    # Ensure everything is nicely cased, even if metadata was messy or missing
    out["title"] = out.apply(
        lambda r: r["title"]
        if pd.notna(r["title"])
        else _prettify_from_code(r["item"]),
        axis=1,
    ).map(_smart_title)

    out["title"] = out["title"].fillna(out["item"])

    # Optional keyword filter for readability
    q = st.text_input("Filter results by keyword (title contains)")
    if q:
        out = out[out["title"].str.contains(q, case=False, na=False)]

    # Order columns neatly
    cols = [
        c
        for c in [
            "item",
            "title",
            "score",
            "enrolls",
            "mean_rating",
            "pop",
            "aff",
            "i2i",
        ]
        if c in out.columns
    ]
    st.dataframe(out[cols], use_container_width=True, height=420)

    # Download
    if not out.empty:
        csv = out[cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV", data=csv, file_name="recommendations.csv", mime="text/csv"
        )

    # 2D visualization of recommended items (optional)
    with st.expander("Visualize similar items (PCA)"):
        if not out.empty:
            top_items = out["item"].head(min(30, len(out))).tolist()
            sub = items[items["item"].isin(top_items)]
            if len(sub) >= 3:
                X = sub[list(c_cols)].values.astype("float32")
                p = PCA(n_components=2, random_state=RS).fit_transform(X)
                dfp = pd.DataFrame(
                    {"x": p[:, 0], "y": p[:, 1], "item": sub["item"].values}
                )
                dfp["title"] = dfp["item"].map(title_map).fillna(dfp["item"])
                st.scatter_chart(dfp, x="x", y="y")
                st.caption(
                    "PCA of recommended items' embeddings (for intuition, not ranking)."
                )
            else:
                st.info("Need at least 3 items to plot.")
        else:
            st.info("No recommendations to visualize yet.")

st.divider()
st.caption(
    "Tips: Upload your own CSVs to override defaults. For cold-start users, select 2–10 liked courses."
)
