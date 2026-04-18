# Detailed EDA Execution Plan — DATATHON 2026

> **Competition:** DATATHON 2026 — The Gridbreaker (VinTelligence — VinUni DS&AI Club)
> **Applicable section:** Part 2 — Visualization & Data Analysis (60/100 points)
> **Methodology:** Combining an iterative workflow (thesis-first vs. data-first) with the task list A1–F3

---

## Table of Contents

1. [Rubric Overview](#rubric-overview)
2. [Phase 0 — Setup](#phase-0--setup-23-hours)
3. [Phase 1 — Shallow EDA](#phase-1--shallow-eda-46-hours)
4. [Phase 2 — Hypothesis Generation & Thesis Selection](#phase-2--hypothesis-generation--thesis-selection-34-hours)
5. [Phase 3 — Targeted Deep Dive](#phase-3--targeted-deep-dive-3-days)
6. [Phase 4 — Stress-Testing](#phase-4--stress-testing-05-days)
7. [Phase 5 — Polish & Report](#phase-5--polish--report-115-days)
8. [7-Day Timeline Summary](#7-day-timeline-summary)
9. [Quick Decision Rules](#quick-decision-rules)
10. [Appendix — Task List A–F](#appendix--task-list-af)

---

## Rubric Overview

| Criterion | Points | Strategic Significance |
|---|---|---|
| Visualization quality | 15 | Every chart must have a title, axis labels, legend, and appropriate chart type |
| Analytical depth | 25 | **Highest weight** — must cover all 4 levels: Descriptive → Prescriptive |
| Business insight | 15 | Specific, quantified, actionable recommendations |
| Creativity & storytelling | 5 | Cross-table connections, unique angle |

**Golden rule:** Every analysis must go through all 4 layers — *What happened → Why → What's next → What to do*. This is where most teams lose points by stopping at Descriptive + Diagnostic.

---

## PHASE 0 — Setup (2–3 hours)

**Goal:** Establish a clean, reproducible working environment.

### Step 0.1: Initialize the repository

Create a public GitHub repo with the following structure:

```
datathon-2026/
├── data/              # raw CSV files (gitignore if too large)
├── notebooks/
│   ├── 01_quick_eda.ipynb
│   ├── 02_hypothesis.ipynb
│   └── 03_deep_dive_<thesis>.ipynb
├── src/               # utility functions
│   ├── loaders.py
│   └── viz_utils.py
├── figures/           # exported charts
├── report/            # LaTeX NeurIPS
└── README.md
```

### Step 0.2: Setup dependencies

- Python environment: `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `statsmodels`, `scikit-learn`, `geopandas` (for Vietnam map)
- Install `scientific-image-generator` skill if not already available
- Set global random seed and matplotlib style

### Step 0.3: Data loader utility

Write `src/loaders.py` with a single function:

```python
def load_all_tables() -> dict[str, pd.DataFrame]:
    """Load 15 CSV files, parse dates, set dtypes"""
```

Used consistently across all notebooks → guarantees consistency.

---

## PHASE 1 — Shallow EDA (4–6 hours)

**Goal:** Understand what the data contains. NO thesis at this stage.
**File:** `01_quick_eda.ipynb`

### Step 1.1: Data quality audit (1.5 hours)

For each of the 15 files:

| Check | Code pattern |
|---|---|
| Shape, dtypes, memory | `df.info()`, `df.shape` |
| Date range | `df['date'].min()`, `.max()` |
| Missing values (%) | `df.isna().mean() * 100` |
| Duplicates | `df.duplicated().sum()` |
| Primary key uniqueness | `df['id'].is_unique` |
| Numeric summary | `df.describe()` |
| Categorical summary | `df.select_dtypes('object').nunique()` |

**Output:** a consolidated data quality table → identify which files have issues to address later.

### Step 1.2: Foreign key validation (1 hour)

Verify all relationships from Table 2 of the problem statement:

- `orders.customer_id` ⊂ `customers.customer_id`?
- `order_items.order_id` ⊂ `orders.order_id`?
- `orders.zip` ⊂ `geography.zip`?
- What proportion of orders have shipments / returns / reviews — does it match the declared cardinality?

Record findings in the notebook, e.g.: *"99.8% of orders have a shipment record, 18% have a review, 4.2% have a return."*

### Step 1.3: Quick descriptive landscape (2 hours)

Run **~15 quick, rough charts** — just to understand the data. These are the essential probes:

**Revenue (input for tasks A1–A3):**
- Monthly revenue line chart (from `sales.csv`)
- Annual revenue bar chart
- Daily revenue histogram

**Customers (input for tasks B1–B4):**
- Monthly new customer line chart (from `signup_date`)
- Orders-per-customer histogram
- Pie charts for acquisition_channel, age_group, gender

**Products (input for tasks C1–C4):**
- Top 20 products by revenue
- Revenue by category
- Histogram of price, COGS, and margin `(price-cogs)/price`

**Promotions (input for tasks D1–D3):**
- Promotion campaign timeline
- Proportion of orders with a promo_id

**Operations (input for tasks E1–E4):**
- Revenue by region (bar chart)
- Shipping days distribution
- Monthly conversion_rate line chart

**Quality (input for tasks F1–F3):**
- Monthly return rate
- Rating distribution
- Order status breakdown

### Step 1.4: Record observations (0.5 hours)

Write a markdown cell at the end of the notebook: *"Surprising things I noticed"*. Examples:

- *"Margin dropped from 48% (2015) to 34% (2022) — a clear trend"*
- *"Streetwear return rate spiked anomalously from 2020"*
- *"70% of promotions are concentrated in Q4 each year"*
- *"Central region contributes only 12% of revenue but has the highest margin"*

**These observations are the raw material for hypothesis generation in Phase 2.**

---

## PHASE 2 — Hypothesis Generation & Thesis Selection (3–4 hours)

**Goal:** Move from "what does the data contain" to "what story will we tell."
**File:** `02_hypothesis.ipynb`

### Step 2.1: Brainstorm 8–10 hypotheses (1 hour)

Based on observations from Step 1.4, write out hypothesis candidates. Each hypothesis follows this format:

```
H_k: [Claim]
  - Preliminary evidence: [From quick EDA]
  - Needs further validation: [Which task from A–F]
  - Potential prescriptive action: [What?]
  - Data tables used: [List them]
```

**Example hypothesis candidates for this dataset:**

| ID | Hypothesis | Related Tasks |
|---|---|---|
| H1 | **Growth-Margin Trade-off:** Revenue growth is driven by discounts while margin is eroding | A1, A3, D1, D2 |
| H2 | **Southern Shift:** Revenue focus is shifting southward but inventory & logistics remain in the North | E1, E2, E3 |
| H3 | **Premium Paradox:** Premium segment has the highest margin but the lowest LTV | B1, B2, C2 |
| H4 | **Size Chart Crisis:** High return rate for size XL — a size chart problem | C3, F1 |
| H5 | **Promo Fatigue:** Later promotions are less effective; customers wait for deals before buying | D1, D2, A2 |
| H6 | **COD Leakage:** COD has a high cancellation rate, causing shipping losses | F3, E2 |
| H7 | **Channel Quality Gap:** Cheap acquisition channels (paid) bring customers who churn quickly | B3, B2 |
| H8 | **Lifecycle Mismatch:** Many SKUs are in decline but are still being restocked | C4, E3 |

### Step 2.2: Quick validation check (1.5 hours)

For each hypothesis, write *quick and dirty* code — ~20–30 lines each — to confirm the evidence:

```python
# H1 quick check
yearly = sales.groupby(sales.Date.dt.year).agg({'Revenue':'sum','COGS':'sum'})
yearly['margin'] = 1 - yearly.COGS/yearly.Revenue
print(yearly)
```

Score each hypothesis on 3 criteria (scale 1–5):

| Hypothesis | Strong evidence? | Business value? | Clear prescriptive? | Total |
|---|---|---|---|---|
| H1 | 5 | 5 | 5 | **15** |
| H2 | 4 | 4 | 4 | 12 |
| H3 | ? | 5 | 4 | ? |
| H4 | 3 | 3 | 4 | 10 |
| ... | | | | |

### Step 2.3: Select the main thesis + sub-plots (1 hour)

**Select 1 main thesis** (highest total score + personal preference) and **2–3 supporting sub-plots**.

**Example — if H1 is selected:**

> **Main thesis — "The Growth Paradox":** The company is achieving impressive revenue growth, but growth quality is deteriorating on 4 fronts: margin compression, declining retention, rising returns, and increasing dependence on promotions.
>
> - **Sub-plot 1:** *Margin erosion* — A1 + A3
> - **Sub-plot 2:** *Promo addiction* — D1 + D2
> - **Sub-plot 3:** *Cohort quality decay* — B2 + B3
> - **Sub-plot 4:** *Operational cost drift* — F1 + E2

These four sub-plots connect 7+ data tables → secures points for creativity.

### Step 2.4: Write the report outline (0.5 hours)

Before diving deeper, draft the 4-page NeurIPS outline:

```
1. Introduction (0.3 pages) — state the thesis
2. Data & Method (0.3 pages) — describe sources and joins
3. Finding 1: Margin Erosion (0.6 pages) — 1–2 charts
4. Finding 2: Promo Addiction (0.6 pages) — 1–2 charts
5. Finding 3: Cohort Decay (0.6 pages) — 1–2 charts
6. Finding 4: Ops Cost Drift (0.4 pages) — 1 chart
7. Strategic Recommendations (0.6 pages)
8. Conclusion (0.2 pages)
9. References + Appendix (not counted toward page limit)
```

→ Target **6–8 charts total**, no more.

---

## PHASE 3 — Targeted Deep Dive (3 days)

**Goal:** Execute the selected tasks, covering all 4 analytical layers: Descriptive → Prescriptive.
**File:** `03_deep_dive_<thesis>.ipynb`

Each sub-plot follows the 4-layer template. Illustrated below for the first sub-plot.

### Day 1 — Sub-plot 1: Margin Erosion (Tasks A1 + A3)

**Tasks A1 + A3 combined, covering all 4 layers:**

#### Descriptive (2 hours)

- Time series: Revenue, COGS, Gross Profit, Margin% by month (48+ data points)
- Revenue decomposition (statsmodels `seasonal_decompose`)
- Summary table: Revenue CAGR, COGS CAGR, change in margin points
- **Chart 1:** Dual-axis — Revenue (left, bar) + Margin% (right, line). Title: *"Revenue grew 4.3x but margin compressed 14 points (2015–2022)"*

#### Diagnostic (2 hours)

- Decompose margin changes by: category, segment, promo vs. non-promo
- Waterfall chart: margin 2015 → margin 2022, broken down by driver
- Determine: is margin declining due to **mix shift** (selling more low-margin categories) or **price pressure** (discounting the same SKUs)?
- **Chart 2:** Waterfall margin decomposition

#### Predictive (1 hour)

- Fit a linear trend to monthly margin
- Extrapolate: when does margin reach X%? When does it hit breakeven?
- Include confidence intervals
- Record findings in narrative text — no separate chart needed (saves space)

#### Prescriptive (1 hour)

- Quantify: if margin had held at 2015 levels, how much more profit would the company have today?
- Recommendations: (1) cap discount rates; (2) shift mix toward high-margin category X; (3) renegotiate COGS with suppliers for the top 10 SKUs
- Estimate impact in numbers

### Day 2 — Sub-plot 2: Promo Addiction (Tasks D1 + D2)

#### Descriptive

- % of revenue from orders with a promotion, by year
- Average discount depth by year
- **Chart 3:** Stacked area — Revenue split between promo-driven and organic, by year

#### Diagnostic

- Incremental lift analysis: promo days vs. baseline (average of non-promo days before/after)
- ROI formula: `(lift_revenue × margin - discount_cost) / discount_cost`
- Interrupted time series around Black Friday: is there a pull-forward effect?
- **Chart 4:** Event study plot — normalized revenue 14 days before/after a promotion

#### Predictive

- At the current trend rate, when will promo cost exceed incremental profit?

#### Prescriptive

- Propose an A/B test: reduce discount depth by 20% on top 3 promotions → predict impact
- Explore shifting from discounts to loyalty points (if data supports this)

### Day 3 — Sub-plots 3 & 4: Cohort Decay & Ops Drift (Tasks B2 + B3 + F1 + E2)

Follow the same 4-layer pattern as above.

**Day 3 morning — Cohort Decay:**
- Cohort retention heatmap (Chart 5)
- Cohort quality broken down by acquisition_channel
- Cohort LTV curves

**Day 3 afternoon — Operational Cost Drift:**
- Return rate + refund_amount over time
- Shipping days × rating × return (correlation analysis)
- Chart 6: Scatter plot of shipping_days vs. 1-star rate, colored by region

---

## PHASE 4 — Stress-Testing (0.5 days)

### Step 4.1: Actively seek counter-evidence (2 hours)

For each finding, proactively ask:

- Is there any category/segment/region that does *not* follow the general trend?
- If COVID years (2020–2021) are excluded, do the conclusions change?
- Is the correlation causal? Are there potential confounders?

Document nuances in the report as intellectual honesty — the judges will appreciate it.

### Step 4.2: Sanity check all numbers (1.5 hours)

- Double-check key figures by computing them a different way
- Ensure consistency across charts (the same metric cannot appear as two different values in two different charts)
- Are any outliers distorting the overall picture?

---

## PHASE 5 — Polish & Report (1–1.5 days)

### Step 5.1: Chart polish (0.5 days)

Using the `scientific-image-generator` skill:

- Consistent color palette (max 5 colors)
- Title = the finding, not a description of the chart (*"Margin collapsed 14 points"* instead of *"Margin over time"*)
- Annotations on charts: arrows, text boxes highlighting key numbers
- Export as PDF vector (for LaTeX) + PNG at 300 DPI (backup)

### Step 5.2: Write the LaTeX NeurIPS report (0.5 days)

One section per finding, following this pattern:

```latex
\subsection{Finding 1: Growth Came at Margin's Cost}

[1 setup sentence]
[Chart with caption stating the finding]
\textbf{Descriptive:} [1 sentence]
\textbf{Diagnostic:} [1–2 sentences]
\textbf{Predictive:} [1 sentence]
\textbf{Prescriptive:} [1–2 sentences with specific numbers]
```

### Step 5.3: Final review (0.5 days)

Final checklist:

- [ ] Every chart has a title, axis labels, legend, and data source
- [ ] Every finding covers all 4 analytical layers
- [ ] Every recommendation has a quantified impact estimate
- [ ] Report is ≤ 4 pages (excluding references and appendix)
- [ ] GitHub repo is public with a complete README
- [ ] Notebooks are fully reproducible (re-running produces identical results)

---

## 7-Day Timeline Summary

| Day | Session | Activity | Deliverable |
|---|---|---|---|
| 1 | Morning | Repo setup + loaders (Phase 0) | `src/loaders.py` |
| 1 | Afternoon | Data quality audit + FK validation (1.1–1.2) | Notebook 01 Part 1 |
| 2 | Morning | Quick descriptive landscape (1.3) | Notebook 01 Part 2 |
| 2 | Afternoon | Observations + brainstorm hypotheses (1.4 + 2.1) | Hypothesis list |
| 3 | Morning | Validation check + select thesis (2.2–2.3) | Notebook 02 |
| 3 | Afternoon | Report outline (2.4) + start deep dive #1 | Outline file |
| 4 | All day | Sub-plot 1: Margin erosion | Charts 1–2 |
| 5 | All day | Sub-plot 2: Promo addiction | Charts 3–4 |
| 6 | Morning | Sub-plot 3: Cohort decay | Chart 5 |
| 6 | Afternoon | Sub-plot 4: Ops drift + stress-testing | Chart 6 |
| 7 | Morning | Chart polish + LaTeX writing | Report v1 |
| 7 | Afternoon | Final review + submission | Final |

---

## Quick Decision Rules

When in doubt during execution, apply these rules:

1. **If you discover an insight stronger than the current thesis** → evaluate it with the 3 criteria from Step 2.2; if it scores higher, pivot decisively (but only before Day 5)
2. **If a task takes more than 4 hours** → cut scope, choose a simpler angle
3. **If a chart does not serve the thesis** → cut it, no matter how good it looks
4. **If a finding lacks a clear Prescriptive action** → find a specific action or drop the finding
5. **If the thesis and data contradict each other on > 30% of evidence** → pivot the thesis

---

## Appendix — Task List A–F

### Group A — Revenue & Growth Analysis

**Task A1. Total revenue time series**
- Plot Revenue by day/week/month/year from `sales.csv` (2012–2022)
- Decompose into trend + seasonality + residual (statsmodels)
- YoY growth, MoM growth, CAGR
- *Predictive:* extrapolate trend → foundation for Part 3 (forecasting)
- *Prescriptive:* identify slow-growth periods → recommend optimal marketing investment windows

**Task A2. Seasonality & holiday effects**
- Revenue heatmap by (month × year) and (day-of-week × week)
- Identify revenue peaks: do they coincide with promotions in `promotions.csv`?
- Quantify lift % for key events: Black Friday, Tết, back-to-school

**Task A3. Gross margin trajectory**
- Compute `(Revenue - COGS) / Revenue` over time
- Compare margin across years — is margin expanding or contracting?
- *Insight:* if margin is declining while revenue grows → a pricing strategy warning signal

### Group B — Customer Analytics

**Task B1. RFM Segmentation**
- Compute Recency, Frequency, Monetary for each customer from `orders` + `order_items`
- K-means or quantile-based segmentation (Champions, Loyal, At-risk, Lost, etc.)
- *Prescriptive:* recommend distinct marketing strategies per segment

**Task B2. Cohort analysis — Retention**
- Define cohorts by `signup_date` (month)
- Retention rate heatmap: cohort × month-since-signup
- Compute average LTV per cohort
- *Diagnostic:* which cohorts retain best? Why? (linked to acquisition_channel?)

**Task B3. Acquisition channel performance**
- Compare `acquisition_channel` values: customer count, LTV, retention, AOV
- Implicit ROI: which channel brings the highest-quality customers?
- *Prescriptive:* rebalance marketing budget across channels

**Task B4. Demographic analysis**
- Revenue / AOV / purchase frequency by `age_group`, `gender`
- Cross-tab: which demographic buys which category most?

### Group C — Product Analytics

**Task C1. Pareto / ABC analysis**
- 80/20 chart: what % of SKUs generate 80% of revenue?
- Classify products as A / B / C
- *Prescriptive:* recommend discontinuing Class C SKUs

**Task C2. Category × Segment performance matrix**
- Revenue, margin, and return rate heatmap by (category × segment)
- Identify "star" cells and "problem" cells

**Task C3. Size & color analysis**
- Return rate by size (linked to Q9 in Part 1)
- Color trends by year — which colors are rising or falling?
- *Diagnostic:* why does size XL have the highest return rate? (size chart inaccuracy?)

**Task C4. Product lifecycle classification**
- Classify products by lifecycle stage: launch → growth → maturity → decline
- Use `inventory.csv` metrics (`sell_through_rate`, `days_of_supply`) combined with sales data

### Group D — Promotion Analytics

**Task D1. Incremental lift analysis**
- Compare revenue on promotion days vs. non-promotion days
- ROI per promotion: `(lift_revenue - discount_cost) / discount_cost`
- Compare percentage-based vs. fixed-amount discounts — which is more effective?

**Task D2. Cannibalization effect**
- Does revenue drop after a promotion ends (pull-forward buying) or rise (new customers attracted)?
- Apply interrupted time series around promotion start/end dates

**Task D3. Stackable promotion analysis**
- When `stackable_flag=1` and `promo_id_2` is not null → is AOV higher?
- *Prescriptive:* recommend a stacking policy

### Group E — Operations Analytics

**Task E1. Geographic heatmap**
- Map of Vietnam by region/city: revenue, AOV, return rate
- Join `orders` + `geography` + `customers`
- *Insight:* which regions are underserved? Where should a new warehouse be opened?

**Task E2. Shipping & delivery performance**
- Distribution of `delivery_date - ship_date` by region
- Correlation between delivery time and review rating / return rate
- *Prescriptive:* define delivery SLAs per region

**Task E3. Inventory health**
- Stockout rate × estimated lost revenue (from `stockout_flag`, `stockout_days`)
- Overstock analysis — how much capital is tied up in excess inventory?
- `days_of_supply` distribution — which SKUs need urgent reordering?

**Task E4. Web traffic → conversion funnel**
- Sessions → unique_visitors → conversion_rate → orders
- Analysis by `traffic_source`: which channel has the best conversion?
- Correlation between bounce_rate and same-day sales

### Group F — Quality & Customer Experience

**Task F1. Return reason deep-dive**
- Cross-tab return_reason × category × size
- Total refund_amount — how much financial leakage is occurring?
- *Prescriptive:* improve size charts, product image quality

**Task F2. Review sentiment & rating analysis**
- Rating distribution by category and segment
- Correlation: rating ↔ return rate ↔ repurchase rate
- From `review_title` → word cloud or keyword analysis (if time permits)

**Task F3. Payment method × cancellation rate**
- Order status breakdown by payment_method
- Does COD have a higher cancellation rate? (linked to Q8 in Part 1)
- *Prescriptive:* COD policy (require deposit, phone verification, etc.)

---

## Closing Notes

- **Core principle:** *thesis-first* leads to confirmation bias; *data-first* leads to getting lost → **iterative** is optimal
- **4-layer rule:** every finding must progress through *Descriptive → Diagnostic → Predictive → Prescriptive*
- **6–8 chart rule:** the 4-page NeurIPS limit forbids chart dumping; every chart must earn its place
- **Storytelling rule:** chart title = the finding, not a description; every recommendation must have a quantified impact

> *"Chart shows X → we found Y → therefore do Z"*
