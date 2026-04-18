import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Google Drive File IDs ─────────────────────────────────────────────────────
DRIVE_IDS = {
    # Pre-computed artifacts — paste IDs after running save_artifacts.py
    "eda_data.pkl":              "1CiYrcdetsu1q4MXV1gBy2XBxFYNZeXlC",
    "model.pkl":                 "1dm4N4Wl3-vij8oohxzJ88sfQ_79wXIaF",
    "apriori_rules.pkl":         "1ZkdYYmrpPzEh7DhWk1MtVBS8ADWy8atV",

    # Raw CSVs (kept as fallback for local dev)
    "orders.csv":                "1beOjK3v7hFLbYvapshc8eDHg8-_1Tlh9",
    "products.csv":              "1yTib9V_yh-v9MSvGviW_TNuxLHLFOKN0",
    "order_products__prior.csv": "1nsvu044dz4S6cjQ3n28xsPbHS_IPcAE4",
    "order_products__train.csv": "16sTwt7_rZkhnYQdZCcFvdE6T0hoBjOhz",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _local_path(name):
    for p in [os.path.join(DATA_DIR, name),
              os.path.join(BASE_DIR, name), name]:
        if os.path.exists(p):
            return p
    return None


def _download(name):
    try:
        import gdown
    except ImportError:
        st.error("gdown not installed. Add gdown>=4.6.0 to requirements.txt.")
        st.stop()

    fid = DRIVE_IDS.get(name, "")
    if not fid or fid.startswith("PASTE_"):
        st.error(f"No Drive ID set for `{name}`. Paste the ID into DRIVE_IDS in data_loader.py")
        st.stop()

    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, name)

    with st.spinner(f"Downloading {name} from Google Drive… (first load only)"):
        gdown.download(f"https://drive.google.com/uc?id={fid}", dest, quiet=False)

    if not os.path.exists(dest):
        st.error(f"Download failed for {name}. Make sure it's shared as 'Anyone with the link'.")
        st.stop()

    return dest


def _get(name):
    path = _local_path(name)
    if path:
        return path
    return _download(name)


# ── EDA loader — returns pre-computed dict, no heavy groupbys at runtime ──────

@st.cache_data(show_spinner="Loading EDA data…")
def load_eda_data():
    """
    Loads eda_data.pkl — a dict of small pre-aggregated dataframes:
        order_kpis        for KPIs and bar charts
        heatmap_df        for heatmap
        top_products      for top products chart
        top_reordered     for top reordered chart
        product_per_order for unique products KPI
    """
    path = _get("eda_data.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# load_combined kept for any other pages that import it
@st.cache_data(show_spinner="Loading data…")
def load_combined():
    return load_eda_data().get("order_kpis")


@st.cache_data(show_spinner="Loading raw datasets…")
def load_raw():
    def read_orders(p):
        df = pd.read_csv(p,
            usecols=["order_id","user_id","eval_set","order_number",
                     "order_dow","order_hour_of_day","days_since_prior_order"],
            dtype={"order_id":"int32","user_id":"int32","eval_set":"category",
                   "order_number":"int16","order_dow":"int8","order_hour_of_day":"int8"})
        df["days_since_prior_order"] = pd.to_numeric(
            df["days_since_prior_order"], errors="coerce").astype("float32")
        return df

    def read_products(p):
        return pd.read_csv(p,
            usecols=["product_id","product_name"],
            dtype={"product_id":"int32","product_name":"str"})

    def read_op(p):
        return pd.read_csv(p,
            usecols=["order_id","product_id","add_to_cart_order","reordered"],
            dtype={"order_id":"int32","product_id":"int32",
                   "add_to_cart_order":"int8","reordered":"int8"})
    try:
        return dict(
            orders         = read_orders(_get("orders.csv")),
            products       = read_products(_get("products.csv")),
            order_products = read_op(_get("order_products__prior.csv")),
            train          = read_op(_get("order_products__train.csv")),
        ), []
    except Exception as e:
        st.error(f"Error loading raw CSVs: {e}")
        return None, [str(e)]


@st.cache_data(show_spinner="Loading association rules…")
def load_apriori_rules():
    path = _get("apriori_rules.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


FEATURE_COLS = [
    "up_order_count","up_reorder_count","up_reorder_rate",
    "up_avg_cart_pos","up_avg_days_between",
    "total_orders","avg_days_between_orders","preferred_day","preferred_hour",
    "product_reorder_rate","product_avg_hour","product_avg_dow",
]


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    path = _get("model.pkl")
    with open(path, "rb") as f:
        a = pickle.load(f)
    return a["model"], a["metrics"], a["feat_imp"], a["threshold"], a["feature_cols"]


@st.cache_data(show_spinner="Engineering features…")
def load_features():
    raw, _ = load_raw()
    if raw is None:
        return None, None, None
    df = (raw["order_products"]
          .merge(raw["orders"][["order_id","user_id","order_number","order_dow",
                                "order_hour_of_day","days_since_prior_order"]],
                 on="order_id", how="left"))

    user_features = (df.groupby("user_id").agg(
        total_orders=("order_number","max"),
        avg_days_between_orders=("days_since_prior_order","mean"),
        preferred_hour=("order_hour_of_day","mean"),
        preferred_day=("order_dow","mean"),
    ).round(2))

    product_features = (df.groupby("product_id").agg(
        product_reorder_rate=("reordered","mean"),
        product_avg_hour=("order_hour_of_day","mean"),
        product_avg_dow=("order_dow","mean"),
    ).round(3))

    up = (df.groupby(["user_id","product_id"]).agg(
        up_order_count=("order_id","count"),
        up_reorder_count=("reordered","sum"),
        up_reorder_rate=("reordered","mean"),
        up_avg_cart_pos=("add_to_cart_order","mean"),
        up_avg_days_between=("days_since_prior_order","mean"),
    ).round(3))

    return user_features, product_features, up
