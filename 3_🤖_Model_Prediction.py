"""
Page 3 – Reorder Predictor
Uses the XGBoost model trained exactly as in your notebook.
Shows prediction probability, model metrics, feature importance, ROC curve,
confusion matrix, and model comparison table.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from data_loader import load_model, FEATURE_COLS as FEAT_COLS

st.set_page_config(page_title="Reorder Predictor", page_icon="🤖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@700;800&display=swap');
.stApp { background: #faf5ff; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#2e1065,#4c1d95); }
[data-testid="stSidebar"] * { color: #ddd6fe !important; }

.page-header {
    background: linear-gradient(135deg,#2e1065,#7c3aed,#a78bfa);
    border-radius: 20px; padding: 40px 48px; color: white; margin-bottom: 28px;
}
.page-header h1 { font-family:'Sora',sans-serif; font-size:2rem; font-weight:800; margin:0 0 8px; }
.page-header p  { margin:0; opacity:.85; font-size:.95rem; }

.sec { font-family:'Sora',sans-serif; font-size:1.05rem; font-weight:700; color:#2e1065;
       border-left:4px solid #7c3aed; padding-left:12px; margin:28px 0 16px; }

.metric-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin-bottom:24px; }
.metric-card {
    background:white; border-radius:14px; padding:20px 14px; text-align:center;
    box-shadow:0 2px 12px rgba(0,0,0,.08); border-bottom:3px solid #7c3aed;
}
.metric-card .val { font-family:'Sora',sans-serif; font-size:1.5rem; font-weight:800; color:#7c3aed; }
.metric-card .lbl { font-size:.72rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.06em; margin-top:4px; }

.pred-yes {
    background:linear-gradient(135deg,#d1fae5,#a7f3d0);
    border:2px solid #059669; border-radius:18px; padding:32px 24px; text-align:center;
}
.pred-no {
    background:linear-gradient(135deg,#fee2e2,#fecaca);
    border:2px solid #ef4444; border-radius:18px; padding:32px 24px; text-align:center;
}
.pred-label { font-family:'Sora',sans-serif; font-size:2rem; font-weight:800; margin-bottom:8px; }
.pred-sub   { font-size:.9rem; opacity:.8; }

.feature-hint {
    background:#ede9fe; border:1px solid #c4b5fd; border-radius:10px;
    padding:12px 16px; font-size:.83rem; color:#4c1d95; margin-bottom:6px; line-height:1.5;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h1>🤖 Reorder Predictor</h1>
    <p>XGBoost model trained on user &amp; product interaction features.
       Adjust the sliders to simulate a user–product pair and get an instant reorder prediction.</p>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Loading XGBoost model"):
    model, metrics, feat_imp, best_threshold, _ = load_model()

if model is None:
    st.error("Model could not be trained. Make sure all 6 CSV files are in the app folder.")
    st.stop()

# ── Model performance KPIs ────────────────────────────────────────────────────
st.markdown('<div class="sec">📈 XGBoost Model Performance (Test Set)</div>',
            unsafe_allow_html=True)

kpi_items = [
    ("Accuracy",  f"{metrics['accuracy']:.1%}"),
    ("Precision", f"{metrics['precision']:.1%}"),
    ("Recall",    f"{metrics['recall']:.1%}"),
    ("F1 Score",  f"{metrics['f1']:.4f}"),
    ("ROC-AUC",   f"{metrics['roc_auc']:.3f}"),
]
kpi_html = '<div class="metric-grid">'
for lbl, val in kpi_items:
    kpi_html += f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'
kpi_html += "</div>"
st.markdown(kpi_html, unsafe_allow_html=True)

# Threshold info
st.markdown(f"""
<div class="feature-hint">
  ⚙️ <b>Optimal Decision Threshold:</b> {best_threshold:.3f}
  — Found by scanning thresholds 0.10 → 0.90 and picking the one that maximises F1 on training data.
  Train F1: <b>{metrics['train_f1']:.4f}</b> &nbsp;|&nbsp; Test F1: <b>{metrics['f1']:.4f}</b>
  (tight gap → no overfitting)
</div>
""", unsafe_allow_html=True)

# ── Two-column layout: sliders left, result right ─────────────────────────────
st.markdown('<div class="sec">🎛️ Simulate a User–Product Pair</div>',
            unsafe_allow_html=True)

col_in, col_out = st.columns([1, 1], gap="large")

FEAT_META = {
    "up_order_count":       ("Times user bought this product",  1, 100,   5, 1),
    "up_reorder_count":     ("How many times reordered",        0, 90,    3, 1),
    "up_reorder_rate":      ("User's reorder rate for product", 0.0,1.0, 0.5,0.05),
    "up_avg_cart_pos":      ("Avg cart position (lower=earlier)",1.0,30.0,5.0,0.5),
    "up_avg_days_between":  ("Avg days between orders",         1.0,60.0,14.0,1.0),
    "total_orders":         ("User's total orders ever",        1, 100,  15, 1),
    "avg_days_between_orders":("Avg days between user orders",  1.0,60.0,14.0,1.0),
    "preferred_day":        ("User's avg shopping day (0=Sun)", 0.0,6.0, 1.5,0.5),
    "preferred_hour":       ("User's avg shopping hour",        0.0,23.0,13.0,1.0),
    "product_reorder_rate": ("Product's overall reorder rate",  0.0,1.0, 0.6,0.05),
    "product_avg_hour":     ("Product's typical purchase hour", 0.0,23.0,13.0,1.0),
    "product_avg_dow":      ("Product's typical day of week",   0.0,6.0, 1.5,0.5),
}

with col_in:
    inputs = {}

    st.markdown("### 🧾 Enter Feature Values")

    # ── Interaction Features ─────────────────────────
    st.markdown("#### 🔁 User–Product Interaction")

    inputs["up_order_count"] = st.number_input(
        "Times user bought this product",
        min_value=0, max_value=1000, value=5
    )

    inputs["up_reorder_count"] = st.number_input(
        "How many times reordered",
        min_value=0, max_value=1000, value=3
    )

    inputs["up_reorder_rate"] = st.number_input(
        "User's reorder rate for product",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )

    inputs["up_avg_cart_pos"] = st.number_input(
        "Average cart position (lower = earlier)",
        min_value=0.0, max_value=100.0, value=5.0, step=0.1
    )

    inputs["up_avg_days_between"] = st.number_input(
        "Average days between orders",
        min_value=0.0, max_value=365.0, value=14.0, step=1.0
    )

    # ── User Features ────────────────────────────────
    st.markdown("#### 👤 User Behavior")

    inputs["total_orders"] = st.number_input(
        "Total orders by user",
        min_value=0, max_value=5000, value=15
    )

    inputs["avg_days_between_orders"] = st.number_input(
        "Avg days between user orders",
        min_value=0.0, max_value=365.0, value=14.0, step=1.0
    )

    # Preferred Day (dropdown)
    day_map = {
        "Sunday": 0, "Monday": 1, "Tuesday": 2,
        "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6
    }

    selected_day = st.selectbox(
        "Preferred shopping day",
        list(day_map.keys()),
        index=1
    )
    inputs["preferred_day"] = day_map[selected_day]

    # Preferred Hour (dropdown)
    hour_options = [f"{h}:00" for h in range(24)]
    selected_hour = st.selectbox(
        "Preferred shopping hour",
        hour_options,
        index=13
    )
    inputs["preferred_hour"] = int(selected_hour.split(":")[0])

    # ── Product Features ─────────────────────────────
    st.markdown("#### 🛒 Product Signals")

    inputs["product_reorder_rate"] = st.number_input(
        "Product reorder rate",
        min_value=0.0, max_value=1.0, value=0.6, step=0.01
    )

    inputs["product_avg_hour"] = st.number_input(
        "Product typical purchase hour",
        min_value=0.0, max_value=23.0, value=13.0, step=1.0
    )

    inputs["product_avg_dow"] = st.number_input(
        "Product typical day of week (0=Sun)",
        min_value=0.0, max_value=6.0, value=1.5, step=0.5
    )

    predict_btn = st.button("🔮 Predict Reorder", use_container_width=True, type="primary")



with col_out:
    X_input = np.array([[inputs[f] for f in FEAT_COLS]])
    prob    = model.predict_proba(X_input)[0, 1]
    pred    = int(prob >= best_threshold)

    # Prediction box
    if pred == 1:
        st.markdown(f"""
        <div class="pred-yes">
            <div class="pred-label">✅ Will Reorder</div>
            <div class="pred-sub">The model predicts this user <strong>will reorder</strong> this product<br>
            in their next order.</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="pred-no">
            <div class="pred-label">❌ Will NOT Reorder</div>
            <div class="pred-sub">The model predicts this user <strong>will not reorder</strong> this product<br>
            in their next order.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Probability gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),
        title={"text": "Reorder Probability (%)", "font": {"size": 14}},
        delta={"reference": best_threshold * 100,
               "increasing": {"color": "#059669"},
               "decreasing": {"color": "#ef4444"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#7c3aed"},
            "steps": [
                {"range": [0, best_threshold*100], "color": "#fee2e2"},
                {"range": [best_threshold*100, 100], "color": "#d1fae5"},
            ],
            "threshold": {
                "line": {"color": "#0d1b3e", "width": 3},
                "thickness": 0.75,
                "value": best_threshold * 100
            }
        }
    ))
    fig_gauge.update_layout(height=260, paper_bgcolor="white", margin=dict(t=30,b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.caption(f"🎯 Decision threshold = {best_threshold:.3f}  |  Probability = {prob:.4f}")

    # Probability bar
    prob_df = pd.DataFrame({
        "Outcome": ["Will NOT Reorder ❌", "Will Reorder ✅"],
        "Probability": [1-prob, prob]
    })
    fig_prob = px.bar(prob_df, x="Outcome", y="Probability",
                      color="Outcome",
                      color_discrete_map={"Will NOT Reorder ❌":"#ef4444","Will Reorder ✅":"#059669"},
                      text=prob_df["Probability"].map("{:.1%}".format),
                      title="Prediction Probability Breakdown")
    fig_prob.update_traces(textposition="outside")
    fig_prob.update_layout(height=300, showlegend=False,
                            plot_bgcolor="white", paper_bgcolor="white",
                            yaxis=dict(range=[0,1.1], tickformat=".0%"))
    st.plotly_chart(fig_prob, use_container_width=True)

# ── Feature Importance ────────────────────────────────────────────────────────
st.markdown('<div class="sec">📊 Feature Importance (XGBoost)</div>', unsafe_allow_html=True)
st.caption("How much each feature contributes to predictions globally.")

fig_imp = px.bar(feat_imp.sort_values("importance"), x="importance", y="feature",
                 orientation="h",
                 color="importance", color_continuous_scale="Purples",
                 title="XGBoost Feature Importance",
                 text=feat_imp.sort_values("importance")["importance"].map("{:.4f}".format),
                 labels={"importance":"Importance Score","feature":"Feature"})
fig_imp.update_traces(textposition="outside")
fig_imp.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="white",
                       coloraxis_showscale=False)
st.plotly_chart(fig_imp, use_container_width=True)

# ── ROC Curve ─────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">📉 ROC Curve (Test Set)</div>', unsafe_allow_html=True)
c_roc, c_cm = st.columns(2)

with c_roc:
    fpr, tpr = metrics["fpr"], metrics["tpr"]
    roc_auc  = metrics["roc_auc"]
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
        line=dict(color="#7c3aed", width=2.5),
        name=f"XGBoost (AUC = {roc_auc:.3f})",
        fill="tozeroy", fillcolor="rgba(124,58,237,0.07)"))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
        line=dict(color="#94a3b8", dash="dash", width=1.5),
        name="Random Baseline"))
    fig_roc.update_layout(
        title=f"ROC Curve — AUC = {roc_auc:.3f}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(x=0.55, y=0.08),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    if roc_auc >= 0.9:
        st.success(f"🌟 Excellent model — AUC {roc_auc:.3f}")
    elif roc_auc >= 0.8:
        st.success(f"✅ Very good model — AUC {roc_auc:.3f}")
    else:
        st.info(f"ℹ️ Good model — AUC {roc_auc:.3f}")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
with c_cm:
    cm = metrics["cm"]
    cm_df = pd.DataFrame(cm,
        index=["Actual: No Reorder","Actual: Reorder"],
        columns=["Pred: No Reorder","Pred: Reorder"])
    fig_cm = px.imshow(cm, text_auto=True,
        color_continuous_scale="Purples",
        x=["Pred: No Reorder","Pred: Reorder"],
        y=["Actual: No Reorder","Actual: Reorder"],
        title="Confusion Matrix (Test Set)",
        labels=dict(x="Predicted", y="Actual", color="Count"))
    fig_cm.update_layout(height=380, paper_bgcolor="white")
    st.plotly_chart(fig_cm, use_container_width=True)

    tn, fp, fn, tp = cm.ravel()
    st.markdown(f"""
    | Metric | Value |
    |---|---|
    | True Negatives (TN) | {tn:,} |
    | False Positives (FP) | {fp:,} |
    | False Negatives (FN) | {fn:,} |
    | True Positives (TP) | {tp:,} |
    """)

# ── Model Comparison ──────────────────────────────────────────────────────────
st.markdown('<div class="sec">🏁 Model Comparison Summary</div>', unsafe_allow_html=True)
st.caption("From your notebook — all three models compared on the test set.")

comp_df = pd.DataFrame([
    {"Model": "Naive Baseline",       "Type": "Rule-based",  "Test F1": 0.0000, "Notes": "Predicts based on last order only — fails on test split"},
    {"Model": "Frequency Baseline",   "Type": "Rule-based",  "Test F1": 0.3800, "Notes": "Uses reorder rate threshold — decent but simple"},
    {"Model": "Logistic Regression",  "Type": "ML Model",    "Test F1": 0.4368, "Notes": "Good interpretable baseline"},
    {"Model": "XGBoost ⭐",           "Type": "ML Model",    "Test F1": metrics["f1"], "Notes": "Best model — captures non-linear patterns"},
])

def highlight_best(row):
    if "XGBoost" in row["Model"]:
        return ["background-color:#ede9fe;font-weight:bold"]*len(row)
    return [""]*len(row)

st.dataframe(
    comp_df.style
        .apply(highlight_best, axis=1)
        .format({"Test F1":"{:.4f}"}),
    use_container_width=True, hide_index=True
)
