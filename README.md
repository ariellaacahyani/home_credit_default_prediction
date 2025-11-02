# Predictive Modeling for Credit Scoring (Home Credit)

**Subtitle:** *Maximizing Loan Approval Through Risk Prediction*

**Author:** Ariella Cahyani
**Presentation Slide:** In repo

---

## 1. Project Background & Objectives (Problem Research)

This project, part of a virtual internship with Home Credit, addresses a core business challenge in lending: balancing risk and opportunity.

1.  **Reducing Credit Loss:** We must accurately identify and reject applicants who are likely to **Default**.
2.  **Maximizing Approval:** We must avoid incorrectly rejecting "good" applicants who are **Capable of Repaying**, as this results in lost revenue.

**Objective (Goal):**
To build a machine learning model (Binary Classification) that provides an accurate credit risk score (probability of default) for each applicant.

**Key Evaluation Metric:**
The dataset is **highly imbalanced** (only 8% Default cases vs. 92% Repaid). Therefore, Accuracy is a misleading metric. Our primary evaluation metric is **AUC-ROC** (Area Under the Receiver Operating Characteristic Curve).

---

## 2. Workflow & Methodology

This project follows a systematic, end-to-end data science workflow:

1.  **Data Cleaning:** Cleaned the main `application_train.csv` (122 features), handled anomalies (`365243 days`), and dropped features with `NaN > 50%`.
2.  **Feature Engineering (FE):** **This was the most critical step.** We aggregated and merged 8 external data files (especially `bureau.csv` and `previous_application.csv`) to "enrich" the main training data.
3.  **EDA & Visualization:** Analyzed the cleaned, enriched data to find actionable business insights.
4.  **Feature Selection (FS):** Used correlation heatmaps to manually remove multicollinear features (>0.9) and prepared for automated selection.
5.  **Data Splitting:** Split the final "rich" data into `train` and `validation` sets (80/20) using `stratify=y` to preserve the imbalanced ratio.
6.  **Data Preprocessing:** Built a robust `ColumnTransformer` `Pipeline` for automated imputation (`Median`/`Mode`), scaling (`RobustScaler`), and encoding (`OneHotEncoder`).
7.  **Modeling & Evaluation (v1):** Tested 3 baseline models (Logistic Regression, Random Forest, LGBM) on all 221 processed features to find the "champion" algorithm.
8.  **Model Selection (v2):** Analyzed the champion's (`LGBM`) `feature_importance_` and filtered the model down to the **Top 50 "Gold" Features**.
9.  **Hyperparameter Tuning (v3):** Fine-tuned the "lean" v2 model (LGBM + 50 features) using `GridSearchCV` to find the optimal settings.
10. **Testing:** Ran the full FE and Preprocessing pipeline on the `application_test.csv` file to prove the model works on new, unseen data.

---

## 3. Key Insights & Business Recommendations (Business Insight)

### Insight 1: Opportunity in "Safe" Segments
* **Finding:** Applicants from the `State servant` (Civil Servant) and `Pensioner` segments have the **lowest default risk** (under 6%).
* **Context:** However, application **volume** from these segments is currently very low (they are in the "Opportunity" quadrant).
* **Recommendation (Action):** Create **targeted marketing campaigns** to acquire more applicants from these segments. We can justify special benefits (like *fast-track approval*) because their low-risk profile is data-driven.

### Insight 2: The "Burden" of Debt (Debt-to-Income Ratio)
* **Finding:** Default risk **increases dramatically** as the "Loan Burden Ratio" (Credit Amount / Income) becomes too heavy.
* **Context:** The risk peaks for applicants borrowing 3-4x their annual income.
* **Recommendation (Action):** Implement a **new underwriting policy** with a **hard cap** on this Debt-to-Income Ratio. Applications exceeding a safe limit (e.g., 3.0) should trigger a mandatory **manual review** by a senior credit analyst.

---

## 4. Final Model & Performance (ML Implementation)

### Model Journey: v1 (Bloated) -> v2 (Lean) -> v3 (Champion)

* **Model v1 (221 Features):** The `LGBMClassifier` was the clear winner with an **AUC of 0.7572**.
* **Model v2 (Top 50 Features):** We **dropped 77% of "junk" features** (where `importance=0`) and **retained 99.8% of the performance** (AUC 0.7553). This proved the model could be far more efficient.
* **Model v3 (Tuned):** The lean v2 (50-feature) model was fine-tuned using `GridSearchCV`.

### Final (Champion) Model

* **Algorithm:** **LightGBM Classifier (`LGBMClassifier`)**
* **Preprocessing:** A `ColumnTransformer` `Pipeline` that automatically handles Imputation (`Median`/`Mode`), Scaling (`RobustScaler`), and Encoding (`OneHotEncoder`). Imbalance is handled by the model's `is_unbalance=True` parameter.
* **Features:** The **Top 50 "Gold" Features** (including `EXT_SOURCE_3`, `AGE`, `CREDIT_INCOME_RATIO`, `BIRO_ACTIVE_COUNT`, `PREV_REFUSED_COUNT`, etc.)
* **Best Hyperparameters:** `{'learning_rate': 0.05, 'max_depth': 20, 'n_estimators': 300}`.
* **Final Performance (CV):** **AUC-ROC = 0.7529** (This is a stable and honest Cross-Validation score).

---

## 5. Business Impact (Business Recommendation)

1.  **Reduce Credit Loss (Protect):** The final model successfully identifies **68%** of all actual defaulters (based on the `Recall` score). This allows the company to proactively **decline high-risk applications** and directly **reduce financial loss**.
2.  **Increase Revenue (Grow):** The model provides data-driven confidence to **increase approval rates** in the "gold" segments (Civil Servants, Pensioners), opening a **new, high-quality revenue stream** from a previously untapped market.
3.  **Improve Operational Efficiency:** The final 77% leaner model (50 features) is **faster** to train, **cheaper** to deploy, and **easier to explain** to regulators and stakeholders.

---

## 6. How to Use This Project

### Tools & References
* **Dataset:** Home Credit Internal Data (Internship Project)  --> Dataset also could access in kaggle
* **Tools:** Python (Pandas, Scikit-learn, Imbalanced-learn, LightGBM, Matplotlib, Seaborn, Joblib)

### Running the Notebook
1.  Clone this repository.
2.  Ensure all libraries listed above are installed.
3.  Place the 10 `.csv` data files from Home Credit in the root directory (or update the file paths).
4.  Run the `FIX_Home_Credit.ipynb` notebook from top to bottom.

### Using the Pre-trained Models
The repository includes all final "assets" (`.joblib` files) for immediate use.

1.  **Load the "Recipe" (Preprocessor):**
    `preprocessor = joblib.load('preprocessor_v1.joblib')`
2.  **Load the "Champion" (Tuned Model):**
    `model = joblib.load('model_lgbm_v3_tuned.joblib')`
3.  **Apply to new data (Must be Feature Engineered first):**
    `data_processed = preprocessor.transform(new_raw_data)`
4.  **Filter to the Top 50 features:**
    `data_final = data_processed[top_50_features]`
5.  **Make predictions:**
    `predictions_proba = model.predict_proba(data_final)`
```eof
