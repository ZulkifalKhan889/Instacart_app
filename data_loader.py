import streamlit as st
import pandas as pd
import pickle
import os

# ── Google Drive File IDs ─────────────────────────────────────────────
DRIVE_IDS = {
    # ✅ ONLY USE ARTIFACTS (NO CSVs)
    "model.pkl":             "1R3rGn0DL_1BQgFe791vWmyXNpd1vPakk",
    "apriori_rules.pkl":     "157iZajxTTOaOYyAR-sjPLqIBwX_xkYmz",
    "order_summary.parquet": "1tJFg5CcEpZu8uvPRLW983-5fQm6nWwL6",
    "product_stats.parquet": "1ANzDrFCqidQOyGbXYDifsgXsAWftZC4Z",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ── Helpers ───────────────────────────────────────────────────────────

def _local_path(name):
    for p in [os.path.join(DATA_DIR, name), os.path.join(BASE_DIR, name), name]:
        if os.path.exists(p):
            return p
    return None


def _download(name):
    import gdown

    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, name)
    fid = DRIVE_IDS[name]

    with st.spinner(f"Downloading {name}..."):
        gdown.download(f"https://drive.google.com/uc?id={fid}", dest, quiet=False)

    return dest


def _get(name):
    path = _local_path(name)
    if path:
        return path
    return _download(name)


# ── FAST LOADERS (ARTIFACT-ONLY) ──────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset...")
def load_combined():
    path = _get("order_summary.parquet")
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading product stats...")
def load_product_stats():
    path = _get("product_stats.parquet")
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading association rules...")
def load_apriori_rules():
    path = _get("apriori_rules.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    path = _get("model.pkl")
    with open(path, "rb") as f:
        artifact = pickle.load(f)

    return (
        artifact["model"],
        artifact["metrics"],
        artifact["feat_imp"],
        artifact["threshold"],
        artifact["feature_cols"],
    )