# DataCo Supply Chain Risk Intelligence
**Optimizing Global Logistics: An Intelligent Early-Warning System to Eliminate Delivery Delays.**

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/4GeeksAcademy/Francisco_Johnny_Marcos_ML_SUPPLY-CHAIN_FP_FEB26_V1)
[![Project Status](https://img.shields.io/badge/Status-Completed-brightgreen)](#)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://francisco-johnny-marcos-ml-supply-chain-7o3d.onrender.com)

## Project Mission
Every late delivery represents a financial penalty and a loss of customer trust. Our team has built a **Proactive Logistics Command Center** that identifies delivery risks at the moment an order is placed, allowing managers to intervene before the shipping process fails.

### The Team
* **Francisco (F)** - Supervised Learning & Lead Logic
* **Johnny (J)** - Data Architecture & Pre-processing
* **Marcos (M)** - Unsupervised Clustering & Strategic UI

---

## Intelligence Architecture: The Dual-Model Approach
Unlike standard predictors, our system utilizes a **Layered Intelligence** strategy to ensure 360-degree visibility:

1.  **Tactical Predictor (Supervised):** A model trained on 180k+ transactions to calculate the exact **Probability of Delay**. It is highly sensitive to critical scheduling conflicts (e.g., Standard Class shipping with a 1-day promise).
2.  **Strategic Profiler (Unsupervised):** A K-Means clustering algorithm that identifies the **Historical Logistics Profile** of a route (Optimal, Standard, or Volatile).

**Conflict Resolution Layer:** Our system includes a logic-override that prioritizes real-time probability over historical clusters, ensuring that "Impossible Promises" are flagged even on typically safe routes.

---

## Model Evaluation & Performance
We evaluated our supervised model based on its ability to identify "Late Risk" before it happens. Since a missed late order (False Negative) costs the company money, we prioritized **Recall** and **F1-Score**.

| **Metric** | **Score** | **Business Interpretation** |
| --- | --- | --- |
| **Accuracy** | 82.4% | "The overall correctness of our "Late" vs "On-Time" predictions. |
| **Recall (Late Class)** | 80.0% | Our ability to catch nearly all at-risk orders before they fail. |
| **Precision** | 87.0% | Ensures we aren't "crying wolf" and flagging too many safe orders. |
| **Historical Delta** | +27.7% | Our model's improvement over the 54.7% baseline (random guessing). |

---

## Key Discoveries & Data Strategy
* **The "Buffer" Secret:** We identified that the #1 predictor of failure is not distance, but the **Scheduling Buffer**. Orders with <2 days of margin have a failure rate nearly double the baseline.
* **Coordinate Integrity:** During EDA, we discovered inaccuracies in raw GPS data. We pivoted to a **Verified Coordinate Lookup** for major hubs (Aachen, Caguas, Barranquilla) to ensure dashboard reliability.
* **High Cardinality:** Managed 500+ unique locations using JSON mapping and factorization to maintain model performance without losing geographical context.

---

## Project Structure

| **File Name** | **Role** | **Model Type** | **Implementation** |
| --- | --- | --- | ---
| supervised_model_final_boost.pkl | **Risk Predictor** | XGBoost | Calculates the probability (%) of a late delivery. |
| unsupervised_kmeans_final.pkl | **Profiler** | K-Means | Segments routes into Strategic Risk Clusters. |
| scaler_WITH_outliers.pkl | **Normalizer** | RobustScaler | Scales real-time user inputs (Price, Benefit) for the model. |
| outliers_dict.jso | **Thresholds** | JSON Map | Stores EDA-defined limits to handle extreme input values. |

---

```text
.
├── README.es.md
├── README.md
├── data
│   ├── interim                                              # JSON mappings & visualization exports
│   │   ├── category_mappings.json
│   │   └── outliers_dict.json
│   ├── processed                                            # Normalized X_train/X_test (With/Without Outliers)
│   │   ├── X_test_WITHOUT_outliers.csv
│   │   ├── X_test_WITHOUT_outliers_norm.csv
│   │   ├── X_test_WITHOUT_outliers_scal.csv
│   │   ├── X_test_WITH_outliers.csv
│   │   ├── X_test_WITH_outliers_norm.csv
│   │   ├── X_test_WITH_outliers_scal.csv
│   │   ├── X_train_WITHOUT_outliers.csv
│   │   ├── X_train_WITHOUT_outliers_WITH_CLUSTERS.csv
│   │   ├── X_train_WITHOUT_outliers_norm.csv
│   │   ├── X_train_WITHOUT_outliers_scal.csv
│   │   ├── X_train_WITH_outliers.csv
│   │   ├── X_train_WITH_outliers_norm.csv
│   │   ├── X_train_WITH_outliers_scal.csv
│   │   ├── df_final_checkpoint.parquet
│   │   ├── y_test.csv
│   │   └── y_train.csv
│   └── raw                                                  # Original DataCo Dataset
│       ├── DataCoSupplyChainDataset.csv
│       └── supply_chain_logistics.db
├── learn.json
├── models                                                   # Production-ready .pkl files. Feature scaling & normalization objects
│   ├── norm_WITHOUT_outliers.pkl
│   ├── norm_WITH_outliers.pkl
│   ├── scaler_WITHOUT_outliers.pkl
│   ├── scaler_WITH_outliers.pkl
│   ├── supervised_model_final_boost.pkl
│   └── unsupervised_kmeans_final.pkl
├── requirements.txt
└── src
    ├── EDA.ipynb                                            # Comprehensive EDA & Heatmap Analysis
    ├── EDA_2.ipynb
    ├── MODELING.ipynb
    ├── STREAMLIT.py                                         # Streamlit Command Center (Wide Layout)
    ├── __pycache__
    │   └── utils.cpython-312.pyc
    └── utils.py                                             # Feature engineering pipeline
```
---

## Technical FAQ (Interview Prep)
Q: **Why are there no Latitude or Longitude columns in the final models?**
A: During the EDA phase, we identified significant inaccuracies and missing values in the raw GPS data. To maintain high Data Integrity, we chose to drop these columns and instead use a "Verified Coordinate Lookup" for our Dashboard. This ensures the visual map is accurate without polluting the model with "noisy" geographic features.

Q: **Why does the app prioritize the Supervised Model over the Unsupervised Clusters?**
A: The K-Means model provides the Strategic Profile (the "personality" of the route), but the Supervised Model provides the Tactical Truth. If a user enters a 1-day promise for a long-haul route, the Supervised model is mathematically more sensitive to that specific risk, so we use it as the final authority for the "Late Risk" status.

Q: **How does the model handle "Outliers" in Price or Benefit?**
A: We utilize a RobustScaler and a predefined outliers_dict.json. This allows the model to remain stable even if a user enters an unusually high order value, preventing "Extreme Data" from skewing the risk prediction.

Q: **What was the logic behind the feature selection?**
A: We removed features with a 1.00 correlation (Multicollinearity), such as "Product Price" vs. "Sales," to reduce model complexity. We focused on the "Scheduled Days" vs. "Shipping Mode" relationship, as our heatmap showed these are the primary drivers of late deliveries.
