import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_loader import load_combined, load_raw

st.set_page_config(page_title="EDA Insights", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@700;800&display=swap');
.stApp { background: #f0f4ff; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0d1b3e,#162554); }
[data-testid="stSidebar"] * { color: #c7d7ff !important; }

.page-header {
    background: linear-gradient(135deg,#1a3a8f,#2563eb,#60a5fa);
    border-radius: 20px; padding: 40px 48px; color: white; margin-bottom: 28px;
}
.page-header h1 { font-family:'Sora',sans-serif; font-size:2rem; font-weight:800; margin:0 0 8px; }
.page-header p  { margin:0; opacity:.85; font-size:.95rem; }

.kpi-wrap { display:grid; grid-template-columns:repeat(6,1fr); gap:14px; margin-bottom:28px; }
.kpi-card {
    background:white; border-radius:14px; padding:22px 16px;
    text-align:center; box-shadow:0 2px 12px rgba(0,0,0,.08);
    border-bottom:4px solid;
}
.kpi-icon { font-size:1.6rem; margin-bottom:6px; }
.kpi-val  { font-family:'Sora',sans-serif; font-size:1.6rem; font-weight:800; line-height:1.1; }
.kpi-lbl  { font-size:.72rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.06em; margin-top:5px; }

.sec { font-family:'Sora',sans-serif; font-size:1.05rem; font-weight:700; color:#0d1b3e;
       border-left:4px solid #2563eb; padding-left:12px; margin:28px 0 16px; }

.chart-card {
    background:white; border-radius:16px; padding:20px;
    box-shadow:0 2px 14px rgba(0,0,0,.07);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h1>📊 EDA &amp; Insights</h1>
    <p>Business KPIs · Order Patterns · Product Rankings · Customer Behaviour · Shopping Heatmap</p>
</div>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_combined()
raw, _ = load_raw()

if df is None:
    st.error("⚠️ CSV files not found.")
    st.stop()

orders         = raw["orders"]
products       = raw["products"]
order_products = raw["order_products"]

days_map = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}
all_days  = list(days_map.values())
all_hours = list(range(24))

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    st.caption("Leave blank to show all data.")

    sel_days = st.multiselect(
        "Filter by Day of Week",
        options=all_days,
        default=[],          # empty = no filter applied
        placeholder="All days (click to filter)…"
    )

    sel_hours = st.multiselect(
        "Filter by Hour of Day",
        options=all_hours,
        default=[],          # empty = no filter applied
        format_func=lambda h: f"{h:02d}:00",
        placeholder="All hours (click to filter)…"
    )

    # Empty selection means "no filter" → use everything
    active_days  = sel_days  if sel_days  else all_days
    active_hours = sel_hours if sel_hours else all_hours

    if sel_days or sel_hours:
        st.markdown("---")
        st.markdown(
            f"**Active filters:**  \n"
            f"{'All days' if not sel_days else ', '.join(sel_days)}  \n"
            f"{'All hours' if not sel_hours else ', '.join(f'{h:02d}:00' for h in sel_hours)}"
        )

# ── Apply filters to the main combined dataframe ──────────────────────────────
filt = df[
    df["day_name"].isin(active_days) &
    df["order_hour_of_day"].isin(active_hours)
]

# Also filter the orders table so KPIs react to filters
filt_order_ids   = filt["order_id"].unique()
orders_filt      = orders[orders["order_id"].isin(filt_order_ids)]
order_prod_filt  = order_products[order_products["order_id"].isin(filt_order_ids)]

# ── KPI Cards (all driven by filtered data) ───────────────────────────────────
total_customers  = orders_filt["user_id"].nunique()
total_orders     = orders_filt["order_id"].nunique()
total_products   = filt["product_id"].nunique()
avg_reorder      = order_prod_filt["reordered"].mean() if len(order_prod_filt) > 0 else 0
basket           = filt.groupby("order_id")["product_id"].count()
avg_basket       = basket.mean() if len(basket) > 0 else 0

# Peak day & hour from filtered data
if len(orders_filt) > 0:
    peak_day  = orders_filt["order_dow"].value_counts().idxmax()
    peak_hour = orders_filt["order_hour_of_day"].value_counts().idxmax()
    peak_label = f"{days_map[peak_day]} {peak_hour:02d}:00"
else:
    peak_label = "N/A"

kpis = [
    ("#2563eb", "👥", f"{total_customers:,}",        "Unique Customers"),
    ("#059669", "🧾", f"{total_orders:,}",           "Total Orders"),
    ("#f59e0b", "📦", f"{total_products:,}",         "Unique Products"),
    ("#ef4444", "🔁", f"{avg_reorder*100:.0f}%",     "Avg Reorder Rate"),
    ("#7c3aed", "🛒", f"{avg_basket:.0f} items",     "Avg Basket Size"),
    ("#0891b2", "⏰", peak_label,                    "Peak Day & Hour"),
]

kpi_html = '<div class="kpi-wrap">'
for color, icon, val, lbl in kpis:
    kpi_html += f"""
    <div class="kpi-card" style="border-color:{color}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-val" style="color:{color}">{val}</div>
        <div class="kpi-lbl">{lbl}</div>
    </div>"""
kpi_html += "</div>"
st.markdown(kpi_html, unsafe_allow_html=True)

if len(filt) == 0:
    st.warning("No data matches the selected filters. Please broaden your selection.")
    st.stop()

# ── Row 1: Orders by Day + Orders by Hour ─────────────────────────────────────
st.markdown('<div class="sec">📅 Order Distribution</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    days_orders = (filt.groupby("order_dow")["order_id"].nunique().reset_index(name="orders"))
    days_orders["pct"] = days_orders["orders"] / days_orders["orders"].sum() * 100
    days_orders["day"] = days_orders["order_dow"].map(days_map)
    days_order_sorted = days_orders.sort_values("order_dow")

    fig = px.bar(days_order_sorted, x="day", y="pct",
                 title="Order Distribution by Day of Week",
                 text=days_order_sorted["pct"].map("{:.1f}%".format),
                 color="pct", color_continuous_scale="Blues",
                 labels={"day": "Day", "pct": "% of Orders"})
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
                      coloraxis_showscale=False, yaxis_range=[0, 30])
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    hour_orders = filt.groupby("order_hour_of_day")["order_id"].nunique().reset_index(name="orders")
    fig2 = px.bar(hour_orders, x="order_hour_of_day", y="orders",
                  title="Total Orders by Hour of Day",
                  text="orders",
                  color="orders", color_continuous_scale="Teal",
                  labels={"order_hour_of_day": "Hour", "orders": "Orders"})
    fig2.update_traces(textposition="outside", marker_line_width=0)
    fig2.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
                       coloraxis_showscale=False)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Heatmap ───────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">🌡️ Shopping Activity Heatmap (Day × Hour)</div>',
            unsafe_allow_html=True)

heat_data = (filt.groupby(["order_dow", "order_hour_of_day"])["order_id"]
             .nunique().reset_index(name="orders"))
heat_data["day"] = heat_data["order_dow"].map(days_map)

pivot = heat_data.pivot(index="order_dow", columns="order_hour_of_day", values="orders").fillna(0)
pivot.index = [days_map[i] for i in pivot.index]

fig_heat = px.imshow(pivot, color_continuous_scale="YlOrRd",
                     title="Orders by Day × Hour",
                     labels=dict(x="Hour of Day", y="Day of Week", color="Orders"),
                     aspect="auto")
fig_heat.update_layout(height=340, paper_bgcolor="white")
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(fig_heat, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Row 3: Top 10 products + Top 10 reordered products ───────────────────────
st.markdown('<div class="sec">🏆 Product Rankings</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)

with c3:
    top10 = (filt.groupby("product_name")["order_id"].count()
             .nlargest(10).reset_index(name="count")
             .sort_values("count"))
    fig3 = px.bar(top10, x="count", y="product_name", orientation="h",
                  title="Top 10 Most Ordered Products",
                  color="count", color_continuous_scale="Blues",
                  text="count", labels={"count": "Orders", "product_name": "Product"})
    fig3.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig3.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                       coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    reordered = (filt[filt["reordered"] == 1].groupby("product_name")["order_id"].count()
                 .nlargest(10).reset_index(name="reordered")
                 .sort_values("reordered"))
    fig4 = px.bar(reordered, x="reordered", y="product_name", orientation="h",
                  title="Top 10 Most Reordered Products",
                  color="reordered", color_continuous_scale="Greens",
                  text="reordered", labels={"reordered": "Reorder Count", "product_name": "Product"})
    fig4.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig4.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                       coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

# ── Row 4: Reorder behaviour pie + Basket size distribution ──────────────────
st.markdown('<div class="sec">🔁 Reorder Behaviour &amp; Basket Size</div>',
            unsafe_allow_html=True)
c5, c6 = st.columns(2)

with c5:
    rr = filt["reordered"].value_counts().reset_index()
    rr.columns = ["Reordered", "Count"]
    rr["Reordered"] = rr["Reordered"].map({1: "Reordered ✅", 0: "New Order 🆕"})
    reorder_pct = filt["reordered"].mean() * 100
    fig5 = px.pie(rr, names="Reordered", values="Count",
                  title=f"Reorder Behaviour ({reorder_pct:.0f}% Reordered)",
                  color_discrete_map={"Reordered ✅": "#059669", "New Order 🆕": "#2563eb"},
                  hole=0.42)
    fig5.update_layout(height=380, paper_bgcolor="white")
    st.plotly_chart(fig5, use_container_width=True)

with c6:
    bsizes = filt.groupby("order_id")["product_id"].count().reset_index(name="basket_size")
    fig6 = px.histogram(bsizes, x="basket_size", nbins=40,
                        title=f"Basket Size Distribution (Avg ≈ {avg_basket:.0f} items)",
                        color_discrete_sequence=["#7c3aed"],
                        labels={"basket_size": "Items per Order"})
    fig6.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig6, use_container_width=True)