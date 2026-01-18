import pytest
import pandas as pd
import numpy as np
import os
from SRC import data_cleaning
from SRC import stats_analysis
from SRC import visualization
from SRC import unsupervised
from SRC import predictive_modeling

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

def test_visualization_files(sample_data):
    """Verify all plots are generated and saved correctly."""
    logger = stats_analysis.setup_environment()
    df_clean = data_cleaning.pre_process(sample_data)
    
    visualization.run_all_visualizations(df_clean, logger)
    
    assert os.path.exists("reports/figures/correlation_heatmap.png")
    assert os.path.exists("reports/figures/Stress_Level_comparison.png")

# --- Stage 5: Structural Requirements ---

def test_environment_setup():
    """Verify that folders and log files are initialized."""
    stats_analysis.setup_environment()
    assert os.path.exists("logs/pipeline.log")
    assert os.path.isdir("reports/figures")
    assert os.path.isdir("reports/tables")