# OULAD Exploratory Data Analysis

An end-to-end exploratory analysis of the [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open-dataset), covering data cleaning, feature engineering, statistical testing, and a simple predictive model of student outcomes.

## Project goal

Understand what factors relate to whether a student passes, fails, withdraws, or earns a distinction in an Open University module — with a particular focus on how student engagement with the Virtual Learning Environment (VLE) relates to outcome.

## Data

Source: Open University's [open dataset page](https://analyse.kmi.open.ac.uk/open-dataset). Seven CSV files, each covering a different piece of the picture:

| File | Description |
|---|---|
| `studentInfo.csv` | Demographics + final result, one row per student-course |
| `studentRegistration.csv` | Registration/unregistration dates |
| `studentAssessment.csv` | Scores each student got on each assessment |
| `assessments.csv` | Assessment metadata (type, deadline, weight) |
| `courses.csv` | Module/presentation info |
| `studentVle.csv` | Daily click counts on VLE resources (~10.6M rows) |
| `vle.csv` | VLE resource metadata |

**Data quality note:** OULAD uses `"?"` as a placeholder for missing values across several columns (`imd_band`, assessment `score`, assessment `date`). These are not automatically recognized as missing by `pandas.read_csv()` and must be explicitly converted before any numeric analysis.

## Setup

```bash
pip install pandas numpy matplotlib scipy scikit-learn
```

Place the CSV files in the project directory (or update the file paths in the notebook), then run the notebook top to bottom.

## Project structure / analysis steps

1. **Load & peek** — read `studentInfo.csv`, inspect shape/columns/dtypes
2. **Missing data** — detect and handle the `"?"` placeholder (`imd_band`, `score`, assessment `date`)
3. **Understand each column** — value counts for categorical columns, `.describe()` for numeric ones, outlier investigation (e.g. `studied_credits`)
4. **Merge tables** — join `studentAssessment` → `assessments` → `studentInfo`, with row-count sanity checks after each merge
5. **Groupby questions** — e.g. outcome by gender
6. **Click activity** — aggregate `studentVle.csv` into per-student-course engagement features (`total_clicks`, `active_days`, `engaged_before_start`)
7. **Visualizations** — boxplot (score by outcome), scatterplot (clicks vs. score, log scale), labeled bar chart (pass rate by education level)
8. **Feature engineering** — submission timeliness (`avg_days_late`), assessment completion rate
9. **Statistical testing** — chi-square test (`engaged_before_start` vs. outcome) and Welch's t-test (clicks, Pass vs. Withdrawn)
10. **Predictive modeling** — logistic regression predicting Pass/Not-Pass, built twice (once with in-course features, once leakage-free) to demonstrate the difference

## Key findings

- **VLE engagement is the strongest signal in the dataset.** Passing students click ~6x more than withdrawn students (1,922 vs. 314 clicks on average), confirmed with a t-test (p ≈ 0).
- **Engaging with the VLE before the course starts** is associated with roughly double the pass rate and half the withdrawal rate (chi-square p ≈ 0, Cramér's V = 0.288 — a real but moderate relationship). Likely reflects underlying motivation rather than a causal effect of early login itself.
- **Assessment completion rate** tracks outcome almost by definition (88% for passers vs. 29% for withdrawn students), since withdrawal typically means stopping partway through.
- **Prior education level** predicts pass rate (29.7% → 65.5% from lowest to highest qualification), though the extreme categories have small sample sizes.
- **Clicks and assessment score are only weakly correlated** (r = 0.29) — engagement predicts *whether* a student finishes far better than *how well* they score.
- **Gender showed almost no relationship** with outcome (39.0% vs. 37.1% pass rate).
- **`imd_band` missingness is not random** — it clusters heavily in North Region (40%) and Ireland (22.5%), likely because England's deprivation index doesn't map cleanly onto other UK regions.
- **A logistic regression confirms the completion-rate leakage concern directly**: a model using in-course features hit 91.3% accuracy, but a leakage-free model (demographics + prior attempts + early engagement only) dropped to a more honest **65% accuracy** — a real but modest predictive signal.

## Caveats

- All relationships found are correlational; no causal claims can be made from this analysis alone.
- Engagement metrics (clicks, active days, completion rate) are themselves *outcomes* of student behavior, not independent causes — demonstrated directly by the 91% vs. 65% accuracy gap between the two models.
- The leakage-free model's 65% accuracy is a more honest ceiling, but still modest — most of the variance in outcome isn't explained by demographics and early engagement alone.
- Sample sizes vary a lot across subgroups (education level, region) — always check group sizes before trusting a percentage.

## Tools used

Python, pandas, NumPy, Matplotlib, SciPy (`chi2_contingency`, `ttest_ind`), scikit-learn (`LogisticRegression`)
