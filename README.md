# CMSC 170: Sleep Stage Classification (Naive Bayes Baseline)

This project builds a Naive Bayes baseline for classifying sleep stages using physiological signals (heart rate and respiratory rate). The dataset contains 500,000 samples across 5 sleep stages.

## 🛠️ Setup Instructions

### 1. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

```powershell
# Create the environment
python -m venv .venv

# Activate the environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
Once the environment is activated, install the required libraries:

```powershell
pip install -r requirements.txt
```

## 📓 Notebooks

The project is divided into two primary workflows:

### 1. `data-cleaning-preprocessing.ipynb`
*   **Purpose:** Initial data ingestion and cleaning.
*   **Key Steps:** Handles missing values, removes duplicates, and ensures the dataset is correctly formatted for analysis.
*   **Output:** Generates the `sleep_stage_sample_cleaned.csv` file used by subsequent steps.

### 2. `data-feature-engineering-categorizing.ipynb`
*   **Purpose:** Prepares data for Naive Bayes and generates the classification baseline.
*   **Key Steps:**
    *   **Feature Selection:** Selects 6 independent features (Heart Rate Mean, SDNN, Slopes, Respiratory Rate Mean, Slopes, and Time).
    *   **Stratified Split:** Performs an 80/20 train/test split.
    *   **Domain-Based Binning:** Categorizes continuous data into discrete clinical bins (e.g., Bradycardia, Normal, Tachycardia).
    *   **Table Generation:** Builds Frequency and Probability tables from the training data.
*   **Output:** Exports categorized CSVs for train/test and provides class prior probabilities.

## 📊 Dataset Overview

*   **Rows:** 500,000
*   **Classes:** 5 (Wake, N1, N2, N3, REM)
*   **Features:** Heart Rate (HR) and Respiratory Rate (RR) metrics derived from physiological sensors.

## 📈 Verification Summary
The pipeline includes automated checks to ensure:
*   Class proportions are preserved after splitting.
*   Binning results in zero empty category-class combinations.
*   Conditional probabilities sum to 1.0 for every class.
