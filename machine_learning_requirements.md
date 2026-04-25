# Part 3 — Sales Forecasting Model

## ▶️ Business Context
You are a data scientist at a Vietnamese fashion e-commerce company.  
The business needs to forecast sales demand accurately to optimize inventory allocation, plan promotions, and manage logistics nationwide.

---

## ▶️ Problem Definition
Forecast the **Revenue** column over the time period in `sales_test.csv`.

Each row in the dataset represents a unique tuple:  
`(Date, Revenue, COGS)` within the time range:

**01/01/2023 – 01/07/2024**

---

## ▶️ Data

| Split | File              | Time Range               |
|------|------------------|-------------------------|
| Train | `sales.csv`       | 04/07/2012 – 31/12/2022 |
| Test  | `sales_test.csv`  | 01/01/2023 – 01/07/2024 |

---

## ▶️ Evaluation Metrics

Submissions are evaluated using three metrics:

### 1. Mean Absolute Error (MAE)
\[
MAE = \frac{1}{n} \sum_{i=1}^{n} |F_i - A_i|
\]

---

### 2. Root Mean Squared Error (RMSE)
\[
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (F_i - A_i)^2}
\]

---

### 3. R² (Coefficient of Determination)
\[
R^2 = 1 - \frac{\sum_{i=1}^{n} (A_i - F_i)^2}{\sum_{i=1}^{n} (A_i - \bar{A})^2}
\]

Where:
- \(F_i\): predicted value  
- \(A_i\): actual value  
- \(\bar{A}\): mean of actual values  

**Notes:**
- MAE measures average absolute error  
- RMSE penalizes larger errors more heavily  
- R² represents the proportion of variance explained by the model  

> ✅ Lower MAE and RMSE are better. Higher R² is better (ideally close to 1).

---

## ▶️ Submission Format

Submit a file named `submission.csv` with the following columns:

**Important:**
- Rows must be in the **exact same order** as in `sample_submission.csv`  
- Do **not** shuffle or remove rows  

---


---

## ▶️ Constraints

### 1. No External Data
All features must be created only from the provided datasets.

---

### 2. Reproducibility
- Submit all source code  
- Set random seeds where necessary  

---

### 3. Explainability
In the report, include a section explaining key factors driving revenue as identified by the model  
(e.g., feature importance, SHAP values, or partial dependence plots).

Translate model insights into **business-friendly explanations**.