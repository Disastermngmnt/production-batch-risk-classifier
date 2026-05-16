# Production Batch Risk Classifier
### Predicting manufacturing cost escalation severity before it compounds

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Type](https://img.shields.io/badge/Type-Multiclass%20Classification-blueviolet)

---

## The Problem

In manufacturing, cost overruns don't announce themselves — they compound quietly until it's too late to act. A machine runs hot for an hour. A supplier delivers late. Scrap accumulates across a shift. By the time the post-batch report flags a Critical overrun, the damage is done.

**This project builds a classifier that predicts how severe a batch's cost escalation will be — using only information available during production — giving operations teams a window to intervene before costs spiral.**

The target has four severity levels: `Normal → Watch → High → Critical`

---

## Results

| Model | Macro F1 | Critical Recall | Balanced Accuracy |
|---|---|---|---|
| **Logistic Regression** | **0.71** | **0.74** | **0.70** |
| Random Forest | 0.68 | 0.69 | 0.68 |
| Decision Tree | 0.62 | 0.63 | 0.61 |

**Why Logistic Regression won:** In structured tabular data with class imbalance, a well-regularised linear model generalised better than a 300-tree ensemble. Random Forest tended to over-index on the majority class (Normal) despite balanced weighting. Logistic Regression caught 74% of Critical batches — the metric that matters most operationally.

**Why not accuracy?** 60% of batches are Normal. A model predicting Normal every time scores 60% accuracy while catching zero at-risk batches. Macro F1 and Critical recall are the right metrics here.

---

## Dataset

| File | Description | Size |
|---|---|---|
| `production_batches.csv` | One row per batch — process metrics, dates, and cost escalation label | 5,050 rows |
| `supplier_quality.csv` | Monthly supplier performance by product category | 3,780 rows |

---

## What Drives Cost Escalation

**Machine Downtime Hours** is the single strongest predictor — highest ANOVA F-statistic and top feature importance. Unplanned stoppages cascade into rework, scrap, and schedule overruns.

**Scrap Rate and Rework Hours** rank second — material wasted and labour re-spent directly inflate batch cost beyond plan.

**Supplier Risk Score and On-Time Delivery Rate** — upstream disruptions from poor suppliers propagate into production cost overruns downstream.

**Batch Size and Planned Cost are not predictive (p > 0.7)** — how big or expensive a batch was planned to be has no relationship with whether it will escalate. Scale is irrelevant. Process failure is everything.

---

## Methodology

### 1 · Data Cleaning & Leakage Removal

Three post-batch columns were identified and removed before any analysis:

| Column Removed | Why |
|---|---|
| `Post_Overrun_Amount` | This IS the cost overrun — directly determines the target |
| `Post_Inspection_Score` | Assigned only after batch completion |
| `Post_Warranty_Claim_Flag` | Filed after delivery — unobservable during production |

Leakage was confirmed empirically: all three columns showed perfect monotonic separation across severity levels.

### 2 · Multi-Table Join Strategy

Production and supplier data were joined on `Supplier_ID + Product_Category + Month` using a **left join** — preserving all production batches even when supplier records were absent, since a missing record is itself a risk signal. Unmatched rows were flagged with `has_supplier_data = 0` and imputed using per-category monthly medians.

### 3 · Statistical Feature Selection

| Test | Applied to | What it confirms |
|---|---|---|
| ANOVA | Numeric features | Do group means differ across the 4 severity levels? |
| Kruskal-Wallis | Numeric features | Non-parametric backup — no normality assumption |
| Chi-Square | Categorical features | Is there a significant association with severity? |
| Cramér's V | Categorical features | How strong is that association? (p-value alone misleads on large samples) |

Non-significant features were excluded before modelling.

### 4 · Feature Engineering

| Feature | Formula | Rationale |
|---|---|---|
| `Cost_Per_Unit` | Planned Cost / Batch Size | Normalises cost across different batch scales |
| `Rework_Per_Unit` | Rework Hours / Batch Size × 1000 | Rework burden per unit produced |
| `Downtime_Ratio` | Downtime / (Rework + 1) | Downtime relative to active rework |
| `Energy_Per_Unit` | Energy kWh / Batch Size | Energy efficiency proxy |
| `Quarter`, `DayOfWeek` | From Production Date | Seasonal and shift-level patterns |
| `Is_Audit_Fail` | 1 if Audit_Flag = Fail | Binary encoding for model consumption |

### 5 · Modelling

- **Split:** 80/20 stratified train/test — class proportions preserved across both sets
- **Preprocessing:** Fitted on training data only, applied to test — no leakage through imputation or scaling
- **Class imbalance:** `class_weight='balanced'` penalises missed Critical batches more heavily
- **Validation:** Held-out test set + 5-fold stratified cross-validation (std < 0.02 across all models — stable generalisation)

---

## Three Actions Operations Can Take Today

**1. Monitor machine downtime in real time.**
If cumulative downtime crosses the Watch→High threshold mid-batch, trigger an engineering review before the shift ends — not after.

**2. Flag high-risk supplier batches at intake.**
Batches from suppliers with Risk Score above the 75th percentile should receive an additional quality check before production begins.

**3. Investigate scrap root causes.**
Scrap Rate is the second most predictive feature and is directly actionable — tooling wear, raw material inconsistency, and operator training are the usual suspects.

---

## Repository Structure

```
production-batch-risk-classifier/
│
├── Submission_Notebook_JD_ADP.ipynb   ← Full analysis notebook
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/Disastermngmnt/production-batch-risk-classifier.git
cd production-batch-risk-classifier
pip install -r requirements.txt
jupyter notebook Submission_Notebook_JD_ADP.ipynb
```

**Dependencies (`requirements.txt`):**
```
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
jupyter
```

---

*Prasidham | [LinkedIn](https://linkedin.com/in/YOUR_HANDLE) | [GitHub](https://github.com/Disastermngmnt)*
