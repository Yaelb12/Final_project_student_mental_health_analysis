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

---


## 🛠 File Structure & Architecture
The project follows a **Modular Clean Architecture** to ensure reproducibility and scientific rigor.

plaintext
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
├── main.py             # Entry point
└── requirements.txt    # Project dependencies

## How to Run
1. **Install Dependencies**: 
   `pip install -r requirements.txt`
2. **Execute Analysis**: 
   `python main.py`
3. **Run Automated Tests**: 
   `python -m pytest tests/`
