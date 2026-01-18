import pandas as pd
import os

def run_risk_prediction_pipeline(df, logger):
    """
    Main pipeline to execute mental health risk prediction analysis.
    Predicts the likelihood of high distress (score 4-5) per major.
    """
    logger.info("Starting Risk Prediction Modeling.")
    
    # Target variables to predict
    targets = {
        'Depression_Score': 'Depression',
        'Anxiety_Score': 'Anxiety',
        'Stress_Level': 'Stress'
    }
    
    # Initializing the report header
    report = "STUDENT MENTAL HEALTH: RISK PREDICTION REPORT\n"
    report += "="*50 + "\n"
    report += "Criteria: 'High Risk' defined as a clinical score of 4 or 5.\n\n"

    # Processing each target variable
    for col, label in targets.items():
        report += _calculate_target_risk(df, col, label)
        
    _save_prediction_report(report, logger)

def _calculate_target_risk(df, column, label):
    """
    Calculates the empirical probability of high risk for a specific target.
    Returns a formatted string section for the report.
    """
    # Create a binary indicator for High Risk (>3)
    df_temp = df.copy()
    df_temp['Is_High_Risk'] = (df_temp[column] > 3).astype(int)
    
    # Group by major and calculate percentage (Probability)
    risk_stats = df_temp.groupby('Course')['Is_High_Risk'].mean() * 100
    risk_stats = risk_stats.sort_values(ascending=False)
    
    # Formatting the visual table section
    section = f"--- PREDICTION PROFILE: {label.upper()} ---\n"
    section += f"{'Academic Major':<25} | {'Risk Chance (%)':<15}\n"
    section += "-"*45 + "\n"
    
    for major, prob in risk_stats.items():
        section += f"{major:<25} | {prob:>12.1f}%\n"
    
    return section + "\n"

def _save_prediction_report(content, logger):
    """
    Saves the final predictive analysis to a text file.
    Ensures the output directory exists.
    """
    output_path = "reports/tables/risk_prediction_report.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(content)
        
    logger.info(f"Predictive Risk Report saved to: {output_path}")