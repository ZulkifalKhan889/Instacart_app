import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import load_eda_data

st.set_page_config(page_title="EDA Insights", page_icon="📊", layout="wide")

# ── Load pre-computed data ────────────────────────────────────────────────────
eda = load_eda_data()
if eda is None:
    st.error("⚠️ EDA data not found.")
    st.stop()

order_kpis        = eda["order_kpis"]         # order_id, user_id, order_dow, order_hour_of_day, day_name, basket_size, reorder_rate
heatmap_df        = eda["heatmap_df"]         # order_dow, order_hour_of_day, day_name, order_count
top_products      = eda["top_products"]       # product_name, order_count
top_reordered     = eda["top_reordered"]      # product_name, reorder_count
product_per_order = eda["product_per_order"]  # order_id, order_dow, order_hour_of_day, unique_products, day_name

# ── Setup ─────────────────────────────────────────────────────────────────────
days_map = {0:"Sun",1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri",6:"Sat"}
all_days  = list(days_map.values())
all_hours = list(range(24))

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    st.caption("Leave blank to show all data.")
    sel_days  = st.multiselect("Day",  options=all_days,  default=[],
                               placeholder="All days — click to filter…")
    sel_hours = st.multiselect("Hour", options=all_hours, default=[],
                               format_func=lambda h: f"{h:02d}:00",
                               placeholder="All hours — click to filter…")
    active_days  = sel_days  if sel_days  else all_days
    active_hours = sel_hours if sel_hours else all_hours

    if sel_days or sel_hours:
        st.markdown("---")
        st.markdown(
            f"**Active filters:**  \n"
            f"{'All days'  if not sel_days  else ', '.join(sel_days)}  \n"
            f"{'All hours' if not sel_hours else ', '.join(f'{h:02d}:00' for h in sel_hours)}"
        )

# ── Filter the pre-computed tables (all tiny, instant) ───────────────────────
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
    st.warning("No data for selected filters.")
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────────────────────
total_customers = filt_orders["user_id"].nunique()
total_orders    = filt_orders["order_id"].nunique()
total_products  = filt_prod_per_order["unique_products"].sum()
avg_reorder     = filt_orders["reorder_rate"].mean()
avg_basket      = filt_orders["basket_size"].mean()
peak_day        = filt_orders["order_dow"].value_counts().idxmax()
peak_hour       = filt_orders["order_hour_of_day"].value_counts().idxmax()

st.title("📊 EDA Insights")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Customers",    f"{total_customers:,}")
c2.metric("Orders",       f"{total_orders:,}")
c3.metric("Products",     f"{int(total_products):,}")
c4.metric("Reorder Rate", f"{avg_reorder*100:.1f}%")
c5.metric("Avg Basket",   f"{avg_basket:.1f}")
c6.metric("Peak",         f"{days_map[peak_day]} {peak_hour:02d}:00")

# ── Orders by Day ─────────────────────────────────────────────────────────────
st.markdown("### 📅 Order Distribution")
c1, c2 = st.columns(2)

with c1:
    day_df = (filt_orders.groupby(["order_dow","day_name"])["order_id"]
              .nunique().reset_index())
    day_df.columns = ["order_dow","day","orders"]
    day_df = day_df.sort_values("order_dow")
    day_df["pct"] = day_df["orders"] / day_df["orders"].sum() * 100
    fig1 = px.bar(day_df, x="day", y="pct",
                  text=day_df["pct"].map("{:.1f}%".format),
                  title="Orders by Day of Week",
                  color="pct", color_continuous_scale="Blues")
    fig1.update_traces(textposition="outside")
    fig1.update_layout(height=360, plot_bgcolor="white",
                       paper_bgcolor="white", coloraxis_showscale=False)
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    hour_df = (filt_orders.groupby("order_hour_of_day")["order_id"]
               .nunique().reset_index())
    hour_df.columns = ["hour","orders"]
    fig2 = px.bar(hour_df, x="hour", y="orders",
                  text="orders", title="Orders by Hour of Day",
                  color="orders", color_continuous_scale="Teal")
    fig2.update_traces(textposition="outside")
    fig2.update_layout(height=360, plot_bgcolor="white",
                       paper_bgcolor="white", coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

# ── Heatmap ───────────────────────────────────────────────────────────────────
st.markdown("### 🌡️ Shopping Activity Heatmap")
pivot = filt_heat.pivot(
    index="order_dow",
    columns="order_hour_of_day",
    values="order_count"
).fillna(0)
pivot.index = [days_map.get(i, i) for i in pivot.index]
fig3 = px.imshow(pivot, color_continuous_scale="YlOrRd",
                 title="Orders by Day × Hour",
                 labels=dict(x="Hour of Day", y="Day", color="Orders"),
                 aspect="auto")
fig3.update_layout(height=340, paper_bgcolor="white")
st.plotly_chart(fig3, use_container_width=True)

# ── Top Products ──────────────────────────────────────────────────────────────
st.markdown("### 🏆 Product Rankings")
c3, c4 = st.columns(2)

with c3:
    top10 = top_products.nlargest(10, "order_count").sort_values("order_count")
    fig4 = px.bar(top10, x="order_count", y="product_name", orientation="h",
                  title="Top 10 Most Ordered Products",
                  color="order_count", color_continuous_scale="Blues",
                  text="order_count")
    fig4.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig4.update_layout(height=380, plot_bgcolor="white",
                       paper_bgcolor="white", coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

with c4:
    top10r = top_reordered.nlargest(10, "reorder_count").sort_values("reorder_count")
    fig5 = px.bar(top10r, x="reorder_count", y="product_name", orientation="h",
                  title="Top 10 Most Reordered Products",
                  color="reorder_count", color_continuous_scale="Greens",
                  text="reorder_count")
    fig5.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig5.update_layout(height=380, plot_bgcolor="white",
                       paper_bgcolor="white", coloraxis_showscale=False)
    st.plotly_chart(fig5, use_container_width=True)

# ── Basket Size Distribution ──────────────────────────────────────────────────
st.markdown("### 🛒 Basket Size Distribution")
fig6 = px.histogram(filt_orders, x="basket_size", nbins=40,
                    title=f"Basket Size Distribution (Avg ≈ {avg_basket:.1f} items)",
                    color_discrete_sequence=["#7c3aed"],
                    labels={"basket_size":"Items per Order"})
fig6.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(fig6, use_container_width=True)
