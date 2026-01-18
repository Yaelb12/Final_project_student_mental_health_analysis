import pandas as pd
import data_cleaning  # Assumes pre_process function exists here
import stats_analysis # Statistical logic
import visualizer     # Graphing logic

def main():
    """
    Main entry point for the Student Mental Health Analysis Pipeline.
    Steps: Setup -> Load -> Clean -> Stats -> Visualize.
    """
    # Step 1: Initialize Logging and Output Directories
    logger = stats_analysis.setup_environment()
    
    # Step 2: Data Pipeline - Load raw CSV and clean it
    logger.info("Initializing Data Pipeline...")
    raw_data = pd.read_csv("st_1.csv")
    df_clean = data_cleaning.pre_process(raw_data)
    
    # Step 3: Analytics - Run T-Tests and ANOVA
    # Defined metrics to analyze across different group types
    metrics = ['Stress_Level', 'Depression_Score', 'Anxiety_Score']
    stats_analysis.run_t_tests(df_clean, metrics, logger)
    stats_analysis.run_anova_and_tukey(df_clean, metrics, logger)
    
    # Step 4: Visualization - Generate and save scientific plots
    visualizer.run_all_visualizations(df_clean, logger)

    logger.info("Pipeline Complete. All reports are available in the 'reports' folder.")

if __name__ == "__main__":
    # Execute the main function
    main()