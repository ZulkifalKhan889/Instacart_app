import streamlit as st

st.set_page_config(
    page_title="Instacart Reorder Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;800&family=Inter:wght@400;500&display=swap');

*, body, .stApp { font-family: 'Inter', sans-serif; }
.stApp { background: #f0f4ff; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b3e 0%, #162554 100%);
}
[data-testid="stSidebar"] * { color: #c7d7ff !important; }
[data-testid="stSidebar"] .stRadio > label { font-size: 15px; font-weight: 600; }
[data-testid="stSidebarNav"] a { border-radius: 8px; }
[data-testid="stSidebarNav"] a:hover { background: rgba(255,255,255,0.08) !important; }

.hero {
    background: linear-gradient(135deg, #0d1b3e 0%, #1a3a8f 50%, #2563eb 100%);
    border-radius: 24px;
    padding: 64px 56px;
    color: white;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 30%;
    width: 400px; height: 400px;
    background: rgba(255,255,255,0.03);
    border-radius: 50%;
}
.hero h1 { font-family: 'Sora', sans-serif; font-size: 2.8rem; font-weight: 800; margin: 0 0 16px; line-height: 1.15; }
.hero p  { font-size: 1.1rem; opacity: .8; max-width: 600px; line-height: 1.7; margin: 0 0 28px; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 100px;
    padding: 6px 18px;
    font-size: .8rem;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
    margin-bottom: 20px;
}

.card-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 20px; margin-bottom: 32px; }
.page-card {
    background: white;
    border-radius: 18px;
    padding: 30px 26px;
    box-shadow: 0 4px 20px rgba(0,0,0,.07);
    border-top: 5px solid;
    transition: transform .2s;
}
.page-card:hover { transform: translateY(-3px); }
.page-card .icon { font-size: 2.2rem; margin-bottom: 14px; }
.page-card h3 { font-family: 'Sora',sans-serif; font-size: 1.05rem; font-weight: 700; margin: 0 0 10px; color: #0d1b3e; }
.page-card p  { font-size: .88rem; color: #64748b; line-height: 1.65; margin: 0; }
.c1 { border-color: #2563eb; }
.c2 { border-color: #059669; }
.c3 { border-color: #7c3aed; }

.stat-row { display: flex; gap: 16px; margin-bottom: 32px; }
.stat-item {
    flex: 1;
    background: white;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,.06);
}
.stat-item .num { font-family:'Sora',sans-serif; font-size: 1.7rem; font-weight: 800; color: #0d1b3e; }
.stat-item .lbl { font-size: .78rem; color: #94a3b8; text-transform: uppercase; letter-spacing:.05em; margin-top: 4px; }

.step-card {
    background: white; border-radius: 14px; padding: 20px 24px;
    display: flex; align-items: flex-start; gap: 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,.06); margin-bottom: 12px;
}
.step-num {
    width: 36px; height: 36px; border-radius: 50%;
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    color: white; font-weight: 800; font-size: .95rem;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.step-card h4 { margin: 0 0 5px; font-weight: 700; font-size: .95rem; color: #0d1b3e; }
.step-card p  { margin: 0; font-size: .85rem; color: #64748b; line-height: 1.55; }
</style>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-badge">🔬 Instacart Basket Analysis</div>
    <h1>Reorder Intelligence<br>Dashboard</h1>
    <p>Explore customer purchasing patterns, uncover product associations with Apriori,
       and predict whether a user will reorder a product in their next order — powered by XGBoost.</p>
</div>
""", unsafe_allow_html=True)

# Dataset stats
st.markdown("""
<div class="stat-row">
    <div class="stat-item"><div class="num">206K+</div><div class="lbl">Unique Customers</div></div>
    <div class="stat-item"><div class="num">3.4M+</div><div class="lbl">Orders (Prior)</div></div>
    <div class="stat-item"><div class="num">49K+</div><div class="lbl">Unique Products</div></div>
    <div class="stat-item"><div class="num">59%</div><div class="lbl">Avg Reorder Rate</div></div>
    <div class="stat-item"><div class="num">0.762</div><div class="lbl">XGBoost AUC</div></div>
</div>
""", unsafe_allow_html=True)

# Page cards
st.markdown("""
<div class="card-grid">
    <div class="page-card c1">
        <div class="icon">📊</div>
        <h3>EDA &amp; Insights</h3>
        <p>KPI tiles, order trends by day &amp; hour, top 10 products, reorder behaviour,
           basket size distribution, and shopping activity heatmap.</p>
    </div>
    <div class="page-card c2">
        <div class="icon">🔗</div>
        <h3>Apriori Rules</h3>
        <p>Interactive rule explorer filtered by support, confidence &amp; lift.
           Bubble scatter chart, top-rules bar chart, network graph, and lift heatmap.</p>
    </div>
    <div class="page-card c3">
        <div class="icon">🤖</div>
        <h3>Reorder Predictor</h3>
        <p>Enter user &amp; product features to get an XGBoost prediction with
           probability gauge, model metrics, feature importance, and ROC curve.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# How to use
st.markdown("### 🚀 How to use this dashboard")
steps = [
    ("Open EDA Insights",    "Navigate to Page 1 via the sidebar. Review the KPI cards at the top, then scroll through order trends, product charts, and the shopping heatmap."),
    ("Explore Apriori Rules","Go to Page 2. Use the sidebar sliders to filter rules by support / confidence / lift, and explore the association network graph."),
    ("Make a Prediction",    "Head to Page 3. Adjust the feature sliders to match a user's behaviour and click Predict — the model returns a probability and explanation instantly.")
    
]
for n, (title, desc) in enumerate(steps, 1):
    st.markdown(f"""
    <div class="step-card">
        <div class="step-num">{n}</div>
        <div><h4>{title}</h4><p>{desc}</p></div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.info("👈  **Select a page from the sidebar to begin.**", icon="ℹ️")
