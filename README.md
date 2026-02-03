# 🧠 Student Mental Health: Predictive & Statistical Analysis
### **A Professional Data Science Pipeline for Psychological Research**

---

## 📋 Project Overview
This research project implements a full, automated data science pipeline to analyze factors affecting student mental health. Developed as part of the **Neuroscience curriculum**, the study explores the relationship between academic majors, lifestyle habits, and psychological distress levels: **Depression, Anxiety, and Stress**.

The project stands out by utilizing both **Supervised Statistical Testing** and a custom **Empirical Risk Prediction Model** to identify vulnerable student populations.

---

## 🔬 Key Research Hypotheses
* **Academic Pressure**: Students in STEM fields (Engineering, Medical, Computing) exhibit significantly higher risk profiles for psychological distress.
* **Risk Predictability**: An individual's academic major serves as a statistically significant predictor for the probability of "High-Risk" mental health outcomes.
* **Internal Consistency**: Depression, Anxiety, and Stress metrics share a high degree of internal correlation, suggesting a single underlying factor of distress.

## 🛠 File Structure & Architecture
The project follows a **Modular Clean Architecture** to ensure reproducibility and scientific rigor.

```bash
Final_project_student_mental_health_analysis/
│
├── DATA/               # Data storage
│   ├── st_1.csv        # Original raw dataset
│   └── clean_data.csv  # Scientifically processed dataset
│
├── SRC/                # Source code (Logic modules)
│   ├── data_cleaning.py# Pre-processing & STEM mapping
│   ├── stats_analysis.py # T-Tests, ANOVA & Environment setup
│   ├── predictive_modeling.py # Risk Prediction Logic
│   ├── unsupervised.py # Factor Analysis (EFA) & KMO testing
│   └── visualization.py# Scientific plotting & Heatmaps
│
├── reports/            # Exported research results
│   ├── figures/        # Scientific PNG charts
│   └── tables/         # Statistical CSV & TXT Reports
│
├── logs/               # Research audit logs
│   └── pipeline.log    # Full system history
│
├── tests/              # Automated QA suite
│   └── test_pipeline.py# 7-Stage Validation Suite
│
├── main.py             # Entry point (Orchestrator)
└── requirements.txt    # Project dependencies

## How to Run
1. **Install Dependencies**: 
   `pip install -r requirements.txt`
2. **Execute Analysis**: 
   `python main.py`
3. **Run Automated Tests**: 
   `python -m pytest tests/`

## 📊 Predictive Risk Profile: Academic Hotspots
Based on our predictive model, we identified specific academic disciplines where the risk of clinical mental health outcomes is significantly elevated.

| Academic Major | Depression Risk | Anxiety Risk | Stress Risk |
| :--- | :---: | :---: | :---: |
| **Computer Science** | **52.2%** | 18.9% | 18.0% |
| **Law** | 20.1% | **48.8%** | 21.6% |
| **Medical** | 19.5% | 20.5% | **49.1%** |
| **Engineering** | 20.0% | 19.9% | 19.0% |
| **Others** | 21.0% | 19.3% | 19.9% |

> **Note:** "High Risk" is defined as a clinical score of 4 or 5 on the standardized assessment scale.

---

## 📉 Anxiety Levels Across Disciplines
The following visualization highlights the significant disparity in anxiety levels, with **Law students** showing the highest average scores compared to all other faculties.

<div align="center">
  <img src="reports/figures/Anxiety_Score_comperarison.png" width="800" alt="Anxiety Score Comparison">
  <br>
  <em>Figure 1: Comparison of Average Anxiety Scores. The red asterisk (*) denotes statistical significance (p < 0.05).</em>
