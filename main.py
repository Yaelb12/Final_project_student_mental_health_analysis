import pandas as pd
import data_cleaning  # Data pre-processing logic
import stats_analysis # Supervised statistical logic
import visualization     # Graphing and visualization logic
import unsupervised   # Unsupervised analysis (EFA) logic
import warnings
import predictive_modeling
# This silences the specific pandas warnings you saw
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    """
    Main entry point for the Student Mental Health Analysis Pipeline.
    Orchestrates the research flow: Setup -> Pre-process -> Statistics -> Visualization -> EFA.
    """
    # Step 1: Initialize Logging and Output Directories
    # This ensures all analysis is documented and folders are ready for files
    logger = stats_analysis.setup_environment()
    
    # Step 2: Data Pipeline - Load raw CSV and clean it
    # We load the raw database and apply mapping to categorical strings
    logger.info("Initializing Data Pipeline...")
    raw_data = pd.read_csv("st_1.csv")
    df_clean = data_cleaning.pre_process(raw_data)
    
    # Step 3: Analytics - Run T-Tests and ANOVA
    # Defined metrics to analyze across STEM and academic courses
    metrics = ['Stress_Level', 'Depression_Score', 'Anxiety_Score']
    stats_analysis.run_t_tests(df_clean, metrics, logger)
    stats_analysis.run_anova_and_tukey(df_clean, metrics, logger)
    predictive_modeling.run_risk_prediction_pipeline(df_clean, logger)
    
    # Step 4: Visualization - Generate and save scientific plots
    # Creates sorted bar charts with significance markers
    visualization.run_all_visualizations(df_clean, logger)

    # Step 5: Unsupervised Analysis - Explore Latent Factor structure (EFA)
    # Checks for correlations and combines indicators into a single distress factor
    visualization.plot_correlation_heatmap(df_clean, metrics, logger)
    unsupervised.run_unsupervised_analysis(df_clean, metrics, logger)

    logger.info("Full Research Pipeline Complete. All results available in 'reports/'.")

if __name__ == "__main__":
    # Execute the research pipeline
    main()