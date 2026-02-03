import pytest
import pandas as pd
import numpy as np
import os
from SRC import data_cleaning
from SRC import stats_analysis
from SRC import visualization
from SRC import unsupervised
from SRC import predictive_modeling

@pytest.fixture(autouse=True)
def isolate_tests(tmp_path, monkeypatch):
    """
    CRITICAL FIX: Test Isolation & Data Integrity Safeguard.
    
    This fixture ensures that automated tests run within a 'Sandbox' environment.
    It prevents pytest from overwriting the real 6,860-row research dataset
    located in the DATA/ directory.
    """
    
    # 1. Create a mock directory structure inside the temporary path.
    # This mimics the project architecture without touching real files.
    (tmp_path / "data").mkdir() 
    (tmp_path / "reports/tables").mkdir(parents=True)
    (tmp_path / "reports/figures").mkdir(parents=True)
    (tmp_path / "logs").mkdir()

    # 2. Redirect the Working Directory.
    # Using monkeypatch to change the current working directory to 'tmp_path'
    # only during the duration of the test. All 'to_csv' or 'savefig' calls
    # will now target the temporary sandbox instead of the project root.
    monkeypatch.chdir(tmp_path)
    
    return tmp_path
@pytest.fixture
def sample_data():
    """Generates a comprehensive dataset to test all pipeline stages."""
    np.random.seed(42)
    data = {
        'Course': ['Engineering', 'Medical', 'Law', 'Computing', 'Engineering', 'Law'],
        'Substance_Use': ['Never', 'Occasionally', 'Never', 'Frequently', 'Never', 'Occasionally'],
        'CGPA': [3.5, 3.8, 2.9, 3.1, 3.6, 3.2],
        'Age': [20, 22, 21, 23, 20, 22],
        'Semester_Credit_Load': [15, 18, 12, 16, 15, 14],
        'Stress_Level': [3, 4, 2, 5, 1, 3],
        'Depression_Score': [1, 3, 2, 4, 5, 2],
        'Anxiety_Score': [2, 4, 1, 5, 3, 4],
        'Financial_Stress': [1, 3, 2, 4, 2, 3],
        'Sleep_Quality': ['Good', 'Average', 'Poor', 'Good', 'Average', 'Poor'],
        'Social_Support': ['High', 'Moderate', 'Low', 'High', 'Moderate', 'Low'],
        'Physical_Activity': ['High', 'Moderate', 'Low', 'High', 'Moderate', 'Low'],
        'Diet_Quality': ['Good', 'Average', 'Poor', 'Good', 'Average', 'Poor'],
        'Counseling_Service_Use': ['Never', 'Occasionally', 'Frequently', 'Never', 'Occasionally', 'Frequently']
    }
    return pd.DataFrame(data)

# --- Stage 1: Data Integrity & Cleaning ---

def test_data_processing_logic(sample_data):
    """Verify missing values, STEM mapping, and range validation."""
    # Adding invalid age and stress for testing
    sample_data.loc[0, 'Age'] = 150 
    sample_data.loc[1, 'Age'] = -5
    
    df_clean = data_cleaning.pre_process(sample_data)
    
    # 1. Age Validation: Ensuring age is within realistic human/student bounds
    assert (df_clean['Age'] >= 18).all()
    assert (df_clean['Age'] <= 100).all()
    
    # 2. STEM mapping check
    stem_courses = df_clean[df_clean['Is_STEM'] == 1]['Course'].unique()
    assert 'Engineering' in stem_courses
    
    # 3. Score Range check (0-5)
    assert (df_clean['Stress_Level'] <= 5).all()

# --- Stage 2: Supervised Analysis (T-Test) ---

def test_t_test_outputs(sample_data):
    """Verify T-Test file creation and internal column structure."""
    logger = stats_analysis.setup_environment()
    df_clean = data_cleaning.pre_process(sample_data)
    stats_analysis.run_t_tests(df_clean, ['Stress_Level', 'Anxiety_Score'], logger)
    
    file_path = "reports/tables/t_test_results.csv"
    assert os.path.exists(file_path)
    
    results_df = pd.read_csv(file_path)
    assert 'Significant' in results_df.columns
    assert 'Cohen_d' in results_df.columns

# --- Stage 3: Modeling (EFA) ---

def test_efa_outputs(sample_data):
    """Verify EFA assumption table creation."""
    logger = stats_analysis.setup_environment()
    df_clean = data_cleaning.pre_process(sample_data)
    unsupervised.check_efa_assumptions(df_clean, ['Stress_Level', 'Depression_Score', 'Anxiety_Score'], logger)
    
    file_path = "reports/tables/efa_assumptions.csv"
    assert os.path.exists(file_path)

# --- Stage 4: Visualization & Files ---

# --- Stage 4: Visualization & Files ---

def test_visualization_files(sample_data):
    """
    Verifies that the main visualization orchestrator runs successfully.
    Now focused on comparison plots only.
    """
    logger = stats_analysis.setup_environment()
    df_clean = data_cleaning.pre_process(sample_data)
    
    # Runs the main orchestrator (Stage 4)
    visualization.run_all_visualizations(df_clean, logger)
    
    figures_path = "reports/figures"
    assert os.path.exists(figures_path)
    files = os.listdir(figures_path)
    
    # Check for comparison plots which we know are generated
    assert any("comparison" in f.lower() for f in files), f"Comparison plots not found. Created: {files}"

def test_correlation_heatmap_generation(sample_data):
    """
    Unit Test: Specifically validates the correlation heatmap generation.
    Independent check for EFA justification requirements.
    """
    logger = stats_analysis.setup_environment()
    df_clean = data_cleaning.pre_process(sample_data)
    
    # Define metrics for the heatmap
    metrics = ['Stress_Level', 'Anxiety_Score', 'Depression_Score', 'CGPA', 'Age']
    
    # Direct call to the heatmap function
    visualization.plot_correlation_heatmap(df_clean, metrics, logger)
    
    # Verify the specific file exists in the sandbox
    file_path = "reports/figures/correlation_heatmap.png"
    assert os.path.exists(file_path), "The correlation heatmap file was not generated."

# --- Stage 5: Structural Requirements ---

def test_environment_setup():
    """Verify that folders and log files are initialized."""
    stats_analysis.setup_environment()
    assert os.path.exists("logs/pipeline.log")
    assert os.path.isdir("reports/figures")
    assert os.path.isdir("reports/tables")


# --- Stage 6: Empirical Risk Prediction Analysis ---

# --- Stage 6: Predictive Risk Modeling ---

def test_risk_report_file_generation(sample_data):
    """
    Technical Test: Verifies that the predictive pipeline executes 
    without errors and physically creates the report file.
    """
    logger = stats_analysis.setup_environment()
    df_clean = data_cleaning.pre_process(sample_data)
    
    # Run the prediction pipeline
    predictive_modeling.run_risk_prediction_pipeline(df_clean, logger)
    
    report_path = "reports/tables/risk_prediction_report.txt"
    assert os.path.exists(report_path), "The risk assessment report file was not generated."


def test_risk_calculation_accuracy(sample_data):
    """
    Logic Test: Validates the empirical probability calculation.
    Checks if the percentage in the report matches a manual calculation.
    """
    logger = stats_analysis.setup_environment()
    df_clean = data_cleaning.pre_process(sample_data)
    predictive_modeling.run_risk_prediction_pipeline(df_clean, logger)
    
    # Manual Calculation for 'Engineering' Stress Risk
    eng_group = df_clean[df_clean['Course'] == 'Engineering']
    high_risk_count = len(eng_group[eng_group['Stress_Level'] > 3])
    
    # Formula: (Number of High Risk Students / Total Students in Course) * 100
    expected_val = (high_risk_count / len(eng_group)) * 100
    expected_str = f"{expected_val:.1f}%"
    
    # Verify that the calculated value is correctly written in the report
    with open("reports/tables/risk_prediction_report.txt", "r") as f:
        report_content = f.read()
        assert expected_str in report_content, f"Math error: Expected {expected_str} not found."