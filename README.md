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
│   └── test_pipeline.py# 8-Stage Validation Suite
│
├── main.py             # Entry point (Orchestrator)
└── requirements.txt    # Project dependencies

```

![Anxiety Comparison](reports/figures/Anxiety_Score_comparison.png)

## How to Run
1. **Install Dependencies**: 
   `pip install -r requirements.txt`
2. **Execute Analysis**: 
   `python main.py`
3. **Run Automated Tests**: 
   `python -m pytest tests/`


## Statistical Analysis and Key Findings

The research implemented a multi-stage statistical pipeline, progressing from broad categorical comparisons to a granular analysis of academic disciplines and a structural evaluation of psychological metrics.

### 1. Categorical Analysis: STEM vs. Non-STEM
The initial stage of the research investigated whether the broad classification into STEM (Science, Technology, Engineering, and Mathematics) and Non-STEM fields serves as a primary driver for mental health distress.
* **Statistical Observation:** While a T-Test revealed a statistically significant difference in distress scores between the groups ($p < 0.05$), the effect size and correlation coefficient were remarkably low ($r \approx 0.18$). 
* **Inference:** This indicates that while STEM students may face distinct pressures, the "STEM" label itself is a weak predictor of clinical outcomes. This finding necessitated a more granular approach, breaking down the data by specific academic majors.

### 2. Discipline-Specific Risk Profiles
Upon analyzing individual majors, the model identified specific "hotspots" where students exhibited significantly higher risks for clinical-level scores (Assessment Score $\ge$ 4). The analysis identified the three most vulnerable disciplines for each metric:

| Assessment Metric | Most Significant Disciplines | Observed Risk Elevation |
| :--- | :--- | :--- |
| **Depression** | Computer Science, Law, Engineering | Significant clustering in high-workload STEM and Professional degrees. |
| **Anxiety** | Law, Medical, Arts & Design | High prevalence in fields with high-stakes licensing and performance requirements. |
| **Stress** | Medical, Biochemistry, Engineering | Strongest correlation with disciplines requiring intensive laboratory and clinical hours. |

### 3. Exploratory Factor Analysis (EFA) & Internal Consistency
To determine if Depression, Anxiety, and Stress represent distinct psychological constructs or a single underlying factor of "Academic Distress," we conducted an Exploratory Factor Analysis (EFA).

* **Pre-test Diagnostics:** * **Kaiser-Meyer-Olkin (KMO) Measure:** $0.72$, indicating "Middling to Good" sampling adequacy for factor analysis.
    * **Bartlett’s Test of Sphericity:** $\chi^2$ significance at $p < 0.001$, confirming that the variables are related and suitable for structure detection.
* **Correlation Matrix:** A Pearson correlation analysis showed strong internal consistency ($r > 0.65$) between the three metrics.
* **Factor Loading:** The EFA confirmed that these three variables load onto a single primary factor, suggesting that for this student population, these symptoms often manifest as a unified psychological response to academic environmental stressors.

### 4. Predictive Modeling Results
Using a custom-built Empirical Risk Model, we calculated the probability of a student reaching a clinical threshold based on their academic profile. The model successfully identified that certain majors (e.g., Computer Science) have a predictive probability for "High-Risk" depression scores exceeding $50\%$.

---

## Predictive Risk Profile: Academic Hotspots

The table below identifies disciplines where the risk of clinical mental health outcomes is significantly elevated based on our predictive model (Clinical Score $\ge$ 4).

| Academic Major       | Depression Risk | Anxiety Risk | Stress Risk |
| :------------------- | :-------------: | :----------: | :---------: |
| **Computer Science** |    **52.2%**    |    18.9%     |    18.0%    |
| **Law**              |      20.1%      |  **48.8%**   |    21.6%    |
| **Medical**          |      19.5%      |    20.5%     |  **49.1%**  |
| **Engineering**      |      20.0%      |    19.9%     |    19.0%    |
| **Others**           |      21.0%      |    19.3%     |    19.9%    |
