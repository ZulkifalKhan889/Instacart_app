# Instacart Market Basket Analysis

A data science project I built to go beyond notebooks — fully deployed, interactive, and solving real engineering problems along the way.

🔗 **[Live App → instacart-app.streamlit.app](https://instacartapp.streamlit.app/EDA_Insights)**  
📁 **[GitHub → Zulkifalkhan889/Instacart_app](https://github.com/Zulkifalkhan889/Instacart_app)**

---

## What this project does

The dataset is from Instacart's open grocery ordering data — 3.4 million orders, 200K+ users, 50K products. I used it to build three things:

- An **EDA dashboard** that lets you explore order patterns by day, hour, and product with interactive filters  
- A **market basket analysis** using the Apriori algorithm to find which products get bought together  
- An **XGBoost model** that predicts whether a user will reorder a specific product in their next order  

Everything runs in a single Streamlit app with a sidebar for navigation.

---

## The part that was actually hard

Getting this to run on Streamlit Cloud's free tier (1 GB RAM) with a 3.1 GB dataset was the real challenge. The naive approach — download CSVs, merge 32 million rows, run groupbys — consistently crashed the server with a silent memory kill.

The fix was to stop doing heavy computation at runtime altogether. I wrote a local script (`save_artifacts.py`) that runs once on my machine, pre-computes every aggregation the dashboard needs, and saves the results as small pickle files (~5 MB each). The deployed app just downloads and displays those — no merging, no groupbys, no crashes.

Local machine Google Drive Streamlit Cloud
───────────── ──────────── ───────────────
save_artifacts.py → upload → eda_data.pkl → download on first load
(runs once, ~15min) model.pkl (~seconds, cached)
apriori_rules.pkl


RAM usage went from ~2.5 GB (crashes) to under 400 MB (stable).

---

## Pages

### 📊 EDA & Insights
Six KPI cards at the top — customers, orders, products, reorder rate, basket size, peak time. Below that: order distribution by day and hour, a day×hour heatmap, top 10 ordered and reordered products, and basket size distribution. All charts respond to the sidebar filters instantly because they're filtering pre-aggregated tables, not raw data.

### 🔗 Association Rules
Apriori on a sample of 5,000 orders across the top 150 products. The output is a filterable table of rules sorted by lift — so you can see things like "customers who buy organic strawberries also tend to buy organic raspberries, with 3.2x lift."

### 🤖 Reorder Prediction Model
XGBoost binary classifier: given a user and a product, will the user reorder it in their next order? Features are engineered at three levels — user behaviour, product popularity, and user-product history. The classification threshold is optimised by sweeping 0.1 to 0.9 and picking the best F1 on the training set. The model page shows accuracy, precision, recall, F1, ROC-AUC, a confusion matrix, and feature importance.

---

## Tech stack

| Component | Tools |
|----------|------|
| App framework | Streamlit |
| Data | Pandas, NumPy |
| Visualisation | Plotly |
| ML | XGBoost, scikit-learn |
| Association rules | MLxtend |
| Data storage | Google Drive via gdown |
| Deployment | Streamlit Community Cloud |

---

## Run it locally

```bash
git clone https://github.com/Zulkifalkhan889/Instacart_app.git
cd Instacart_app
pip install -r requirements.txt
streamlit run app.py
```

The app will automatically download the data files from Google Drive on first run. If you want to use your own copy of the data, place these files in the project root or a `data/` subfolder:

```
orders.csv
products.csv
order_products__prior.csv
order_products__train.csv
```

---

## Dataset

[Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis) from Kaggle.

| File | Description |
|------|------------|
| `orders.csv` | One row per order — user, timing, order sequence |
| `products.csv` | Product names and department/aisle IDs |
| `order_products__prior.csv` | Products in each prior order (32M rows) |
| `order_products__train.csv` | Products in the training order — used as labels |

The CSVs total ~3.1 GB and are not stored in this repo. They live on Google Drive and are pulled down automatically by `data_loader.py`.

---

## Model details

**Task:** binary classification — will user *u* reorder product *p* in their next order?

### Features engineered at three levels:

- **User level** — how many orders they've placed, average days between orders, preferred shopping hour and day  
- **Product level** — overall reorder rate, average hour and day it gets ordered  
- **User–product level** — how many times this user has ordered this product, their personal reorder rate for it, average cart position, average days between reorders  

**Training:**  
70/30 split on users with `eval_set == 'train'`. Class imbalance handled with `scale_pos_weight`. Threshold sweep from 0.1 to 0.9 to maximise F1.

---

## Project structure

```
Instacart_app/
├── app.py                   — landing page
├── data_loader.py           — all data loading, caching, model loading
├── requirements.txt
├── README.md
└── pages/
    ├── 1_📊_EDA_Insights.py
    ├── 2_🔗_Apriori_Rules.py
    └── 3_🤖_ML_Model.py
```

---

## Author

**Zulkifal Khan**  
[GitHub](https://github.com/Zulkifalkhan889)
