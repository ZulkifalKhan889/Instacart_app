
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from data_loader import load_apriori_rules

st.set_page_config(page_title="Apriori Rules", page_icon="🔗", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@700;800&display=swap');
.stApp { background: #f0fff8; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#052e16,#064e3b); }
[data-testid="stSidebar"] * { color: #a7f3d0 !important; }

.page-header {
    background: linear-gradient(135deg,#052e16,#059669,#34d399);
    border-radius: 20px; padding: 40px 48px; color: white; margin-bottom: 28px;
}
.page-header h1 { font-family:'Sora',sans-serif; font-size:2rem; font-weight:800; margin:0 0 8px; }
.page-header p  { margin:0; opacity:.85; font-size:.95rem; }

.sec { font-family:'Sora',sans-serif; font-size:1.05rem; font-weight:700; color:#052e16;
       border-left:4px solid #059669; padding-left:12px; margin:28px 0 16px; }

.kpi-row { display:flex; gap:14px; margin-bottom:24px; flex-wrap:wrap; }
.kpi-pill {
    background:white; border-radius:12px; padding:14px 20px;
    flex:1; min-width:150px; text-align:center;
    box-shadow:0 2px 10px rgba(0,0,0,.07); border-top:3px solid #059669;
}
.kpi-pill .val { font-family:'Sora',sans-serif; font-size:1.5rem; font-weight:800; color:#059669; }
.kpi-pill .lbl { font-size:.75rem; color:#6b7280; text-transform:uppercase; letter-spacing:.05em; }

.lift-high { background:#d1fae5 !important; color:#065f46 !important; }
.lift-mid  { background:#fef9c3 !important; color:#713f12 !important; }
.lift-low  { background:#fee2e2 !important; color:#7f1d1d !important; }

.insight-box {
    background:#ecfdf5; border:1px solid #6ee7b7; border-radius:12px;
    padding:16px 20px; margin-bottom:16px; font-size:.9rem; color:#065f46; line-height:1.6;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h1>🔗 Apriori Association Rules</h1>
    <p>Top 200 products · 5,000 sampled orders · min_support=0.002 · filter by support / confidence / lift</p>
</div>
""", unsafe_allow_html=True)

# ── Load rules ────────────────────────────────────────────────────────────────
with st.spinner("Running Apriori algorithm"):
    rules = load_apriori_rules()

if rules is None:
    st.warning("Could not generate rules. Make sure the CSV files are present and mlxtend is installed.")
    st.info("Run:  `pip install mlxtend`")
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Rule Filters")
    min_sup  = st.slider("Min Support",     float(rules["support"].min()),
                          float(rules["support"].max()),
                          float(rules["support"].quantile(0.1)), step=0.001, format="%.3f")
    min_conf = st.slider("Min Confidence",  float(rules["confidence"].min()), 1.0,
                          float(rules["confidence"].quantile(0.2)), step=0.01, format="%.2f")
    min_lift = st.slider("Min Lift",        float(rules["lift"].min()),
                          float(rules["lift"].max()), 1.0, step=0.1, format="%.1f")
    top_n    = st.slider("Max rules to show", 5, min(100,len(rules)), 30)

filt = rules[
    (rules["support"]    >= min_sup) &
    (rules["confidence"] >= min_conf) &
    (rules["lift"]       >= min_lift)
].head(top_n).reset_index(drop=True)

# ── KPI pills ─────────────────────────────────────────────────────────────────
n_rules  = len(filt)
avg_lift = filt["lift"].mean() if n_rules else 0
avg_conf = filt["confidence"].mean() if n_rules else 0
avg_sup  = filt["support"].mean() if n_rules else 0

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-pill"><div class="val">{n_rules}</div><div class="lbl">Filtered Rules</div></div>
  <div class="kpi-pill"><div class="val">{avg_lift:.2f}</div><div class="lbl">Avg Lift</div></div>
  <div class="kpi-pill"><div class="val">{avg_conf:.1%}</div><div class="lbl">Avg Confidence</div></div>
  <div class="kpi-pill"><div class="val">{avg_sup:.4f}</div><div class="lbl">Avg Support</div></div>
  <div class="kpi-pill"><div class="val">{len(rules)}</div><div class="lbl">Total Rules</div></div>
</div>
""", unsafe_allow_html=True)

if n_rules == 0:
    st.warning("No rules match. Try lowering the sliders.")
    st.stop()

# ── Insight box ───────────────────────────────────────────────────────────────
top_rule = filt.sort_values("lift", ascending=False).iloc[0]
st.markdown(f"""
<div class="insight-box">
  🏆 <b>Strongest Rule:</b>
  <b>{top_rule['antecedents']}</b> → <b>{top_rule['consequents']}</b>
  &nbsp;|&nbsp; Lift: <b>{top_rule['lift']:.2f}</b>
  &nbsp;|&nbsp; Confidence: <b>{top_rule['confidence']:.1%}</b>
  &nbsp;|&nbsp; Support: <b>{top_rule['support']:.4f}</b>
  <br>
  Customers who buy <b>{top_rule['antecedents']}</b> are
  <b>{top_rule['lift']:.1f}×</b> more likely to also buy <b>{top_rule['consequents']}</b> than a random shopper.
</div>
""", unsafe_allow_html=True)

# ── Rules table ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec">📋 Filtered Rules Table</div>', unsafe_allow_html=True)

display = filt[["antecedents","consequents","support","confidence","lift","conviction"]].copy()
display.index = display.index + 1

def color_lift(v):
    if v >= 3:   return "background-color:#d1fae5;color:#065f46"
    if v >= 1.5: return "background-color:#fef9c3;color:#713f12"
    return "background-color:#fee2e2;color:#7f1d1d"

styled = (display.style
    .format({"support":"{:.4f}","confidence":"{:.4f}","lift":"{:.3f}","conviction":"{:.3f}"})
    .map(color_lift, subset=["lift"]))
st.dataframe(styled, use_container_width=True, height=300)

# ── Top rules bar ─────────────────────────────────────────────────────────────
st.markdown('<div class="sec">🏆 Top Rules by Lift</div>', unsafe_allow_html=True)
top15 = filt.nlargest(15,"lift").copy()
top15["rule"] = top15["antecedents"] + "  →  " + top15["consequents"]
fig_bar = px.bar(top15.sort_values("lift"), x="lift", y="rule", orientation="h",
                 color="confidence", color_continuous_scale="Teal",
                 title="Top 15 Rules (sorted by Lift)",
                 text=top15.sort_values("lift")["lift"].map("{:.2f}".format),
                 labels={"lift":"Lift","rule":"Rule","confidence":"Confidence"})
fig_bar.update_traces(textposition="outside")
fig_bar.update_layout(height=500, plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(fig_bar, use_container_width=True)

# ── Lift heatmap ──────────────────────────────────────────────────────────────
if n_rules >= 4:
    st.markdown('<div class="sec">🌡️ Lift Heatmap (Antecedent × Consequent)</div>',
                unsafe_allow_html=True)
    top_pivot = filt.nlargest(40,"lift")
    pivot = top_pivot.pivot_table(index="antecedents",columns="consequents",
                                   values="lift",aggfunc="max").fillna(0)
    fig_h = px.imshow(pivot, color_continuous_scale="YlOrRd",
                      title="Max Lift between Products", aspect="auto",
                      height=max(350, len(pivot)*30))
    fig_h.update_layout(paper_bgcolor="white")
    st.plotly_chart(fig_h, use_container_width=True)

# ── Interpretation guide ──────────────────────────────────────────────────────
st.markdown('<div class="sec">📖 How to read the metrics</div>', unsafe_allow_html=True)
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.info("**Support** = how often the pair appears in all orders.\n\n`support = 0.003` means ~15 of 5,000 orders contained both items together.")
with col_b:
    st.success("**Confidence** = if customer buys antecedent, probability they also buy consequent.\n\n`conf = 0.42` → 42% of buyers of item A also bought item B.")
with col_c:
    st.warning("**Lift** = how much more likely the pair is vs random.\n\n`lift = 2.5` → 2.5× stronger than random chance. Lift > 1 = positive association.")
