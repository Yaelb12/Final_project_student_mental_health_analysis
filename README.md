# Student Mental Health: Predictive & Statistical Analysis

## Project Overview
This research project implements a full data science pipeline to analyze factors affecting student mental health. The study explores the relationship between academic majors, lifestyle habits, and psychological distress levels (Depression, Anxiety, and Stress). 

The project stands out by utilizing both **Supervised Statistical Testing** and **Empirical Risk Prediction Models** to identify vulnerable student populations.

---

## Key Research Hypotheses
1. **Academic Pressure**: Students in STEM fields (Engineering, Medical, Computing) exhibit higher risk profiles for psychological distress.
2. **Risk Predictability**: An individual's academic major serves as a significant predictor for the probability of "High-Risk" mental health outcomes.
3. **Internal Consistency**: Depression, Anxiety, and Stress metrics share a high degree of internal correlation, suggesting a single underlying factor of distress.

---

## Methodology & Pipeline Stages

### 1. Data Preprocessing & Cleaning
* **Outlier Handling**: Used the Interquartile Range (IQR) method to remove unrealistic values in Age and CGPA.
* **Feature Engineering**: Created the `Is_STEM` binary indicator to allow for high-level comparisons across disciplines.
* **Categorical Encoding**: Mapped qualitative survey data (e.g., "Good", "Poor") to numerical ordinal scales (1-3) using the `.map()` function for statistical reliability.

### 2. Statistical Inference
* **Hypothesis Testing**: Conducted Independent T-Tests and ANOVA (including Tukey Post-Hoc) to find significant differences between groups.
* **Exploratory Factor Analysis (EFA)**: Validated the structural integrity of the mental health survey data by checking KMO and Bartlett's assumptions.

### 3. Predictive Risk Modeling
* **Model Type**: Empirical Probability Model (Risk Stratification).
* **Logic**: The model calculates the likelihood of a student scoring in the "High-Risk" range (4 or 5 out of 5) based on their specific academic major.
* **Formula**: $P(\text{High Risk} | \text{Major}) = \frac{\text{Count of High Risk Students in Major}}{\text{Total Students in Major}}$

### 4. Automated Quality Assurance (QA)
* The pipeline is protected by **5 Integrated Test Stages** using `pytest`:
    * **Data Logic**: Validates cleaning and range constraints.
    * **Stats Integrity**: Ensures T-test results include critical columns like `Cohen_d`.
    * **Visuals & Env**: Verifies that all reports and figures are physically saved to the correct directories.

---

## File Structure
* `main.py`: The central execution script.
* `data_cleaning.py`: Handles preprocessing and logical validation.
* `predictive_modeling.py`: Generates the Risk Probability Report.
* `stats_analysis.py`: Executes supervised tests and environment setup.
* `unsupervised.py`: Performs EFA and dimensionality analysis.
* `visualization.py`: Produces scientific heatmaps and bar charts.
* `tests/test_pipeline.py`: Automated testing suite.

---

## How to Run
1. **Install Dependencies**: 
   `pip install -r requirements.txt`
2. **Execute Analysis**: 
   `python main.py`
3. **Run Automated Tests**: 
   `python -m pytest tests/`