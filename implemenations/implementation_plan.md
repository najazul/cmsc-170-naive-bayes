# Naive Bayes Frequency/Probability Tables — Sleep Stage Dataset

## Goal

Build Naive Bayes **frequency/probability tables** (like the "Play Tennis" example) for a sleep stage classification dataset. This is the baseline dataset — no oversampling/undersampling yet.

---

## Dataset: 500,000 rows, 5 classes

| Code | Label    | Count   | Proportion |
|------|----------|---------|------------|
| 0    | Wake     | 154,414 | 30.9%      |
| 1    | N1_Light | 23,423  | 4.7%       |
| 2    | N2_Light | 199,933 | 40.0%      |
| 3    | N3_Deep  | 55,187  | 11.0%      |
| 5    | REM      | 67,043  | 13.4%      |

> [!NOTE]
> Classes are **imbalanced** (N1 is only 4.7%). We acknowledge this but disregard it for now — this baseline is important for future over/undersampling comparisons.

---

## Pipeline (Agreed)

```
Full Dataset (500K rows)
    │
    ├── 1. Select 6 feature columns + target
    ├── 2. 80/20 stratified split
    │       ├── x_train, y_train (400K rows)
    │       └── x_test, y_test (100K rows)
    ├── 3. Domain-based binning (boundaries defined from training data)
    │       ├── x_train_categorized.csv
    │       ├── y_train.csv
    │       ├── x_test_categorized.csv
    │       └── y_test.csv
    └── 4. Build frequency/probability tables (from training data only)
```

**Output: 4 CSV files + frequency/probability tables**

---

## Columns to Drop

| Column | Reason |
|--------|--------|
| `hr_rmssd_5` | Redundant with `hr_sdnn_5` (r=0.95) |
| `hr_rr_ratio` | Derived from `hr_mean / rr_mean` — violates independence |
| `hr_rr_product` | Derived from `hr_mean × rr_mean` — violates independence |
| `rr_sd_5` | Moderate correlation with `rr_mean` (r=0.42); less independent |
| `night_id` | Identifier, not a feature |
| `split` | Disregarded per user instruction |
| `stage_name` | Disregarded per user instruction |

---

## 6 Selected Features (Agreed)

| # | Feature | Type | Why |
|---|---------|------|-----|
| 1 | `hr_mean` | Physiological | Core heart rate; independent from respiratory features |
| 2 | `hr_sdnn_5` | Physiological | Heart rate variability; captures stability |
| 3 | `hr_slope_3` | Physiological | HR trend; most independent feature (all corr ≤ 0.20) |
| 4 | `rr_mean` | Physiological | Core respiratory rate |
| 5 | `rr_slope_3` | Physiological | RR trend; low correlation with all others (≤ 0.28) |
| 6 | `minutes_since_start` | Temporal | Sleep architecture is time-dependent (deep sleep early, REM late); independent from physiological signals (corr ≤ 0.12) |

**Target: `sleep_stage`** (0, 1, 2, 3, 5)

---

## Domain-Based Binning Strategy

> [!IMPORTANT]
> Bin boundaries are determined using **domain knowledge / clinical standards**, but validated against the training data distribution to ensure no bins are empty.

### 1. `hr_mean` — Heart Rate (bpm)

| Category | Range | Clinical Basis |
|----------|-------|----------------|
| Bradycardia | < 60 bpm | Below normal resting HR |
| Normal | 60–100 bpm | Standard adult resting HR range |
| Tachycardia | > 100 bpm | Elevated HR (arousal/movement) |

*Data check: median=65.96, 25th=59.07, 75th=75.70 → most data falls in Normal, reasonable spread across all 3*

### 2. `hr_sdnn_5` — HR Variability (SDNN over 5 epochs)

| Category | Range | Domain Basis |
|----------|-------|--------------|
| Low | < 1.0 | Very stable HR — typical of deep sleep |
| Moderate | 1.0–3.0 | Normal variability — light sleep/REM |
| High | > 3.0 | High variability — arousals, transitions, wake |

*Data check: median=1.38, 25th=0.73, 75th=2.75 → good spread across all 3 bins*

### 3. `hr_slope_3` — HR Trend (slope over 3 epochs)

| Category | Range | Domain Basis |
|----------|-------|--------------|
| Decreasing | < −0.5 | HR declining — relaxation/deepening sleep |
| Stable | −0.5 to 0.5 | No significant HR change |
| Increasing | > 0.5 | HR rising — arousal/lightening sleep |

*Data check: 25th=−0.50, 75th=0.47 → ~50% of data in Stable, balanced spread*

### 4. `rr_mean` — Respiratory Rate (breaths/min)

| Category | Range | Clinical Basis |
|----------|-------|----------------|
| Low | < 15 | Slow breathing — deep sleep pattern |
| Normal | 15–25 | Standard adult respiratory range during sleep |
| High | > 25 | Elevated breathing — arousals, REM, movement |

*Data check: median=18.68, 25th=14.88, 75th=27.08 → good spread*

### 5. `rr_slope_3` — RR Trend (slope over 3 epochs)

| Category | Range | Domain Basis |
|----------|-------|--------------|
| Decreasing | < −0.5 | Breathing slowing — transition to deeper sleep |
| Stable | −0.5 to 0.5 | No significant breathing change |
| Increasing | > 0.5 | Breathing quickening — arousal/lightening |

*Data check: 25th=−0.61, 75th=0.62 → ~50% in Stable, balanced*

### 6. `minutes_since_start` — Time in Session

| Category | Range | Sleep Architecture Basis |
|----------|-------|------------------------|
| Early | 0–180 min | First 3 hours — predominantly N3 deep sleep cycles |
| Mid | 180–360 min | Hours 3–6 — mixed N2 and increasing REM |
| Late | > 360 min | Hours 6+ — predominantly REM and light sleep |

*Data check: 25th=133.5, 50th=266, 75th=400 → good spread across all 3*

---

## Table Output Format

For each of the 6 features, we produce a table like this (matching the photo):

**Example — `hr_mean` (Heart Rate)**

|              | Wake  | N1   | N2   | N3   | REM  | P(Wake)  | P(N1)  | P(N2)  | P(N3)  | P(REM) |
|--------------|-------|------|------|------|------|----------|--------|--------|--------|--------|
| Bradycardia  | ...   | ...  | ...  | ...  | ...  | .../n₀   | .../n₁ | .../n₂ | .../n₃ | .../n₅ |
| Normal       | ...   | ...  | ...  | ...  | ...  | .../n₀   | .../n₁ | .../n₂ | .../n₃ | .../n₅ |
| Tachycardia  | ...   | ...  | ...  | ...  | ...  | .../n₀   | .../n₁ | .../n₂ | .../n₃ | .../n₅ |
| **Total**    | n₀    | n₁   | n₂   | n₃   | n₅   |          |        |        |        |        |

Where `P(class) = count_in_bin_for_class / total_for_class`

---

## Proposed Changes

### [NEW] `split_and_categorize.py`

A Python script (using the venv) that:

1. Loads `sleep_stage_sample_cleaned.csv`
2. Selects the 6 features + `sleep_stage`
3. Performs 80/20 stratified split using `sklearn.model_selection.train_test_split`
4. Applies domain-based binning to both train and test
5. Saves 4 CSVs:
   - `x_train_categorized.csv` (6 categorized feature columns)
   - `y_train.csv` (sleep_stage column)
   - `x_test_categorized.csv` (6 categorized feature columns)
   - `y_test.csv` (sleep_stage column)
6. Prints frequency/probability tables for each feature (from training data)

### [NEW] `naive_bayes_tables.py` (optional, separate)

If you prefer the table generation as a standalone script for the notebook/report.

---

## Verification Plan

### Automated
- Verify x_train + x_test = 500,000 rows
- Verify class proportions are preserved in both train and test (stratification check)
- Verify no bin is empty for any class
- Verify P(category|class) values sum to 1.0 per column
- Verify all row totals match class counts

### Manual
- Review tables together for interpretability
