import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import load_eda_data

st.set_page_config(page_title="EDA Insights", page_icon="📊", layout="wide")

# ── Styling ───────────────────────────────────────────────────────────────────
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
.page-header h1 {
    font-family:'Sora',sans-serif; font-size:2rem;
    font-weight:800; margin:0 0 8px;
}
.page-header p { margin:0; opacity:.85; font-size:.95rem; }

.kpi-wrap {
    display:grid; grid-template-columns:repeat(6,1fr);
    gap:14px; margin-bottom:28px;
}
.kpi-card {
    background:white; border-radius:14px; padding:22px 16px;
    text-align:center; box-shadow:0 2px 12px rgba(0,0,0,.08);
    border-bottom:4px solid;
}
.kpi-icon { font-size:1.6rem; margin-bottom:6px; }
.kpi-val  {
    font-family:'Sora',sans-serif; font-size:1.6rem;
    font-weight:800; line-height:1.1;
}
.kpi-lbl  {
    font-size:.72rem; color:#94a3b8;
    text-transform:uppercase; letter-spacing:.06em; margin-top:5px;
}

.sec {
    font-family:'Sora',sans-serif; font-size:1.05rem;
    font-weight:700; color:#0d1b3e;
    border-left:4px solid #2563eb; padding-left:12px; margin:28px 0 16px;
}

.chart-card {
    background:white; border-radius:16px; padding:20px;
    box-shadow:0 2px 14px rgba(0,0,0,.07);
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>📊 EDA &amp; Insights</h1>
    <p>Business KPIs · Order Patterns · Product Rankings · Customer Behaviour · Shopping Heatmap</p>
</div>
""", unsafe_allow_html=True)

# ── Load pre-computed data ────────────────────────────────────────────────────
eda = load_eda_data()
if eda is None:
    st.error("⚠️ EDA data not found.")
    st.stop()

order_kpis        = eda["order_kpis"]
heatmap_df        = eda["heatmap_df"]
top_products      = eda["top_products"]
top_reordered     = eda["top_reordered"]
product_per_order = eda["product_per_order"]

days_map = {0:"Sun",1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri",6:"Sat"}
all_days  = list(days_map.values())
all_hours = list(range(24))

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    st.caption("Leave blank to show all data.")

    sel_days = st.multiselect(
        "Filter by Day of Week",
        options=all_days, default=[],
        placeholder="All days — click to filter…"
    )
    sel_hours = st.multiselect(
        "Filter by Hour of Day",
        options=all_hours, default=[],
        format_func=lambda h: f"{h:02d}:00",
        placeholder="All hours — click to filter…"
    )

    active_days  = sel_days  if sel_days  else all_days
    active_hours = sel_hours if sel_hours else all_hours

    if sel_days or sel_hours:
        st.markdown("---")
        st.markdown(
            f"**Active filters:**  \n"
            f"{'All days'  if not sel_days  else ', '.join(sel_days)}  \n"
            f"{'All hours' if not sel_hours else ', '.join(f'{h:02d}:00' for h in sel_hours)}"
        )

# ── Apply filters ─────────────────────────────────────────────────────────────
filt_orders = order_kpis[
    order_kpis["day_name"].isin(active_days) &
    order_kpis["order_hour_of_day"].isin(active_hours)
]
filt_heat = heatmap_df[
    heatmap_df["day_name"].isin(active_days) &
    heatmap_df["order_hour_of_day"].isin(active_hours)
]
filt_prod_per_order = product_per_order[
    product_per_order["day_name"].isin(active_days) &
    product_per_order["order_hour_of_day"].isin(active_hours)
]

if filt_orders.empty:
    st.warning("No data for selected filters. Please broaden your selection.")
    st.stop()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
total_customers = filt_orders["user_id"].nunique()
total_orders    = filt_orders["order_id"].nunique()
total_products  = int(filt_prod_per_order["unique_products"].sum())
avg_reorder     = filt_orders["reorder_rate"].mean()
avg_basket      = filt_orders["basket_size"].mean()
peak_day        = filt_orders["order_dow"].value_counts().idxmax()
peak_hour       = filt_orders["order_hour_of_day"].value_counts().idxmax()
peak_label      = f"{days_map[peak_day]} {peak_hour:02d}:00"

kpis = [
    ("#2563eb", "👥", f"{total_customers:,}",    "Unique Customers"),
    ("#059669", "🧾", f"{total_orders:,}",       "Total Orders"),
    ("#f59e0b", "📦", f"{total_products:,}",     "Unique Products"),
    ("#ef4444", "🔁", f"{avg_reorder*100:.0f}%", "Avg Reorder Rate"),
    ("#7c3aed", "🛒", f"{avg_basket:.1f} items", "Avg Basket Size"),
    ("#0891b2", "⏰", peak_label,                "Peak Day & Hour"),
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

# ── Order Distribution ────────────────────────────────────────────────────────
st.markdown('<div class="sec">📅 Order Distribution</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    day_df = (filt_orders.groupby(["order_dow","day_name"])["order_id"]
              .nunique().reset_index())
    day_df.columns = ["order_dow","day","orders"]
    day_df = day_df.sort_values("order_dow")
    day_df["pct"] = day_df["orders"] / day_df["orders"].sum() * 100

    fig1 = px.bar(day_df, x="day", y="pct",
                  title="Order Distribution by Day of Week",
                  text=day_df["pct"].map("{:.1f}%".format),
                  color="pct", color_continuous_scale="Blues",
                  labels={"day":"Day","pct":"% of Orders"})
    fig1.update_traces(textposition="outside", marker_line_width=0)
    fig1.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
                       coloraxis_showscale=False, yaxis_range=[0,30])
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    hour_df = (filt_orders.groupby("order_hour_of_day")["order_id"]
               .nunique().reset_index())
    hour_df.columns = ["hour","orders"]

    fig2 = px.bar(hour_df, x="hour", y="orders",
                  title="Total Orders by Hour of Day",
                  text="orders",
                  color="orders", color_continuous_scale="Teal",
                  labels={"hour":"Hour","orders":"Orders"})
    fig2.update_traces(textposition="outside", marker_line_width=0)
    fig2.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
                       coloraxis_showscale=False)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Heatmap ───────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">🌡️ Shopping Activity Heatmap (Day × Hour)</div>',
            unsafe_allow_html=True)

pivot = filt_heat.pivot(
    index="order_dow",
    columns="order_hour_of_day",
    values="order_count"
).fillna(0)
pivot.index = [days_map.get(i, i) for i in pivot.index]

fig3 = px.imshow(pivot, color_continuous_scale="YlOrRd",
                 title="Orders by Day × Hour",
                 labels=dict(x="Hour of Day", y="Day of Week", color="Orders"),
                 aspect="auto")
fig3.update_layout(height=340, paper_bgcolor="white")
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(fig3, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Product Rankings ──────────────────────────────────────────────────────────
st.markdown('<div class="sec">🏆 Product Rankings</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)

with c3:
    top10 = top_products.nlargest(10, "order_count").sort_values("order_count")
    fig4 = px.bar(top10, x="order_count", y="product_name", orientation="h",
                  title="Top 10 Most Ordered Products",
                  color="order_count", color_continuous_scale="Blues",
                  text="order_count",
                  labels={"order_count":"Orders","product_name":"Product"})
    fig4.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig4.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                       coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

with c4:
    top10r = top_reordered.nlargest(10, "reorder_count").sort_values("reorder_count")
    fig5 = px.bar(top10r, x="reorder_count", y="product_name", orientation="h",
                  title="Top 10 Most Reordered Products",
                  color="reorder_count", color_continuous_scale="Greens",
                  text="reorder_count",
                  labels={"reorder_count":"Reorder Count","product_name":"Product"})
    fig5.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig5.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                       coloraxis_showscale=False)
    st.plotly_chart(fig5, use_container_width=True)

# ── Basket Size Distribution ──────────────────────────────────────────────────
st.markdown('<div class="sec">🛒 Basket Size Distribution</div>',
            unsafe_allow_html=True)

fig6 = px.histogram(filt_orders, x="basket_size", nbins=40,
                    title=f"Basket Size Distribution (Avg ≈ {avg_basket:.1f} items)",
                    color_discrete_sequence=["#7c3aed"],
                    labels={"basket_size":"Items per Order"})
fig6.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white")
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(fig6, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)
