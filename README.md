# 🛒 Instacart Market Basket Analysis Dashboard

An end-to-end, interactive data science project built with **Streamlit** that explores the [Instacart Market Basket dataset](https://www.kaggle.com/c/instacart-market-basket-analysis) — covering exploratory analysis, customer behaviour, association rule mining, and a machine learning model that predicts product reorders.


---

## 📌 Project Overview

This dashboard was built to demonstrate a full data science pipeline — from raw data ingestion and EDA all the way to a deployed, interactive ML-powered application. The dataset contains over **3 million orders** from more than **200,000 Instacart users**.

---

## 🧩 Pages & Features

### 📊 EDA & Insights
- Business KPIs — unique customers, total orders, avg basket size, reorder rate, peak day & hour
- Order distribution by day of week and hour of day
- Shopping activity heatmap (Day × Hour)
- Top 10 most ordered and most reordered products
- Reorder behaviour breakdown and basket size distribution
- **Interactive sidebar filters** — filter all charts and KPIs by day of week and hour

### 🔁 Association Rules (Apriori)
- Market basket analysis using the Apriori algorithm
- Discover which products are frequently bought together
- Filterable rules table by support, confidence, and lift

### 🤖 ML Model — Reorder Prediction
- XGBoost binary classifier trained to predict whether a user will reorder a product
- Feature engineering at user, product, and user-product level
- Threshold-optimised for F1 score
- Model evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
- ROC curve, Confusion Matrix, and Feature Importance charts

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Dashboard & UI | Streamlit |
| Data Manipulation | Pandas, NumPy |
| Visualisation | Plotly |
| Machine Learning | XGBoost, Scikit-learn |
| Association Rules | MLxtend |
| Data Storage | Google Drive (via gdown) |
| Deployment | Streamlit Community Cloud |

---

## 📁 Project Structure

```
instacart-dashboard/
│
├── app.py                        # Main entry point
├── data_loader.py                # Data loading, feature engineering, model training
├── requirements.txt              # Python dependencies
│
└── pages/
    ├── eda_insights.py           # EDA & Insights page
    ├── association_rules.py      # Apriori / Market Basket page
    └── ml_model.py               # XGBoost model & evaluation page
```

> **Note:** The raw CSV files (~3.1 GB) are not stored in this repo. They are hosted on Google Drive and downloaded automatically at runtime via `gdown`.

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/Zulkifalkhan889/Instacart_app.git
cd Instacart_app
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your data**

Place the following CSV files in the project root or a `data/` subfolder:
```
orders.csv
products.csv
order_products__prior.csv
order_products__train.csv
```
Or they will be downloaded automatically from Google Drive on first run.

**4. Run the app**
```bash
streamlit run app.py
```

---

## 📦 Dataset

- **Source:** [Instacart Market Basket Analysis — Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis)
- **Size:** ~3.1 GB across 4 CSV files
- **Scale:** 3.4M orders · 206K users · 50K products

| File | Description |
|---|---|
| `orders.csv` | Order metadata per user |
| `products.csv` | Product names and IDs |
| `order_products__prior.csv` | Products in prior orders |
| `order_products__train.csv` | Products in training orders (labels) |

---

## 🧠 ML Model Details

**Task:** Binary classification — will a user reorder a given product in their next order?

**Features engineered across 3 levels:**

| Level | Features |
|---|---|
| User | Total orders, avg days between orders, preferred hour & day |
| Product | Reorder rate, avg order hour, avg order day |
| User–Product | Order count, reorder count, reorder rate, avg cart position, avg days between |

**Model:** XGBoost with `scale_pos_weight` to handle class imbalance  
**Threshold:** Optimised on training F1 score across a sweep from 0.1 to 0.9

---

## ⚙️ Deployment Notes

The app is deployed on **Streamlit Community Cloud**. Since the dataset is ~3.1 GB, CSVs are hosted on Google Drive. On first load, `data_loader.py` downloads them automatically using `gdown` and caches them with `@st.cache_data` so subsequent visitors load instantly.

---

## 👤 Author

**Zulkifal khan**  
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/Zulkifalkhan889) · [Email](mailto:zulkifalkhan126@email.com)
