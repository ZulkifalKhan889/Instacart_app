import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.express as px

from data_loader import load_combined  # ✅ ONLY THIS

st.set_page_config(page_title="EDA Insights", page_icon="📊", layout="wide")

# ── Load data (FAST parquet) ───────────────────────────────────────────
df = load_combined()

if df is None:
    st.error("⚠️ Data not found.")
    st.stop()

# ── Setup ─────────────────────────────────────────────────────────────
days_map = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}
all_days  = list(days_map.values())
all_hours = list(range(24))

# ── Sidebar filters ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Filters")

    sel_days = st.multiselect(
        "Day of Week", options=all_days, default=[]
    )

    sel_hours = st.multiselect(
        "Hour", options=all_hours, default=[]
    )

    active_days  = sel_days if sel_days else all_days
    active_hours = sel_hours if sel_hours else all_hours

# ── Apply filters ─────────────────────────────────────────────────────
filt = df[
    df["day_name"].isin(active_days) &
    df["order_hour_of_day"].isin(active_hours)
]

if len(filt) == 0:
    st.warning("No data for selected filters.")
    st.stop()

# ── KPIs (now directly from df) ───────────────────────────────────────
total_customers = filt["user_id"].nunique()
total_orders    = filt["order_id"].nunique()
total_products  = filt["product_id"].nunique()
avg_reorder     = filt["reordered"].mean()
avg_basket      = filt.groupby("order_id")["product_id"].count().mean()

peak_day  = filt["order_dow"].value_counts().idxmax()
peak_hour = filt["order_hour_of_day"].value_counts().idxmax()

st.title("📊 EDA Insights")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Customers", total_customers)
col2.metric("Orders", total_orders)
col3.metric("Products", total_products)
col4.metric("Reorder Rate", f"{avg_reorder*100:.1f}%")
col5.metric("Avg Basket", f"{avg_basket:.1f}")

st.markdown(f"**Peak:** {days_map[peak_day]} at {peak_hour}:00")

# ── Charts ───────────────────────────────────────────────────────────

# Orders by day
day_df = filt.groupby("order_dow")["order_id"].nunique().reset_index()
day_df["day"] = day_df["order_dow"].map(days_map)

fig1 = px.bar(day_df, x="day", y="order_id", title="Orders by Day")
st.plotly_chart(fig1, use_container_width=True)

# Orders by hour
hour_df = filt.groupby("order_hour_of_day")["order_id"].nunique().reset_index()

fig2 = px.bar(hour_df, x="order_hour_of_day", y="order_id", title="Orders by Hour")
st.plotly_chart(fig2, use_container_width=True)

# Heatmap
heat = filt.groupby(["order_dow","order_hour_of_day"])["order_id"].nunique().reset_index()
pivot = heat.pivot(index="order_dow", columns="order_hour_of_day", values="order_id").fillna(0)

fig3 = px.imshow(pivot, title="Heatmap (Day vs Hour)")
st.plotly_chart(fig3, use_container_width=True)

# Top products
top = filt.groupby("product_name")["order_id"].count().nlargest(10).reset_index()

fig4 = px.bar(top, x="order_id", y="product_name", orientation="h",
              title="Top Products")
st.plotly_chart(fig4, use_container_width=True)

# Reordered products
reord = filt[filt["reordered"]==1].groupby("product_name")["order_id"].count().nlargest(10).reset_index()

fig5 = px.bar(reord, x="order_id", y="product_name", orientation="h",
              title="Top Reordered Products")
st.plotly_chart(fig5, use_container_width=True)