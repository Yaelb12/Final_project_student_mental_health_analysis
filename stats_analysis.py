import pandas as pd
import numpy as np
import logging
import os
import sys
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def setup_environment():
    """
    Sets up the research environment by creating output folders 
    and initializing a dual-stream logger (file and console).
    """
    # Create the folder structure for results if it doesn't exist
    os.makedirs("reports/tables", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    
    # Configure the logging format and target files
    log_format = "%(asctime)s %(levelname)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler("reports/analysis_log.txt", encoding='utf-8'),
            logging.StreamHandler(sys.stdout) # Output to terminal
        ]
    )
    return logging.getLogger(__name__)

def calculate_cohen_d(group1, group2):
    """
    Calculates Cohen's d to measure the effect size (magnitude of difference)
    between two independent samples.
    """
    # Get sample sizes and variances
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Calculate pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Return the d-value (Difference of means divided by pooled std)
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def run_t_tests(df, variables, logger):
    """
    Performs Independent Samples T-Tests to compare STEM and Non-STEM students.
    Saves a summary CSV with statistics, p-values, and effect sizes.
    """
    logger.info("--- STARTING T-TEST ANALYSIS (STEM vs Non-STEM) ---")
    t_results = []
    
    for var in variables:
        # Split data into groups based on the 'Is_STEM' flag
        stem = df[df['Is_STEM'] == 1][var]
        non_stem = df[df['Is_STEM'] == 0][var]
        
        # Calculate t-statistic and p-value
        t_stat, p_val = stats.ttest_ind(stem, non_stem)
        d_val = calculate_cohen_d(stem, non_stem)
        
        # Append results to the summary list
        t_results.append({
            'Variable': var,
            'T-Statistic': round(t_stat, 3),
            'P-Value': round(p_val, 4),
            'Cohen_d': round(d_val, 3),
            'Significant': p_val < 0.05
        })
    
    # Convert list to DataFrame and save to the tables folder
    pd.DataFrame(t_results).to_csv("reports/tables/t_test_results.csv", index=False)
    logger.info("T-Test summary saved successfully.")

def run_anova_and_tukey(df, variables, logger):
    """
    Runs One-Way ANOVA across different academic courses.
    If ANOVA is significant (p < 0.05), it proceeds to Tukey HSD post-hoc test.
    """
    logger.info("--- STARTING ANOVA ANALYSIS (By Course) ---")
    courses = df['Course'].unique()
    
    for var in variables:
        # Prepare groups for ANOVA test
        groups = [df[df['Course'] == c][var] for c in courses]
        f_stat, p_val = stats.f_oneway(*groups)
        
        if p_val < 0.05:
            logger.info(f"Significant variance detected in {var}. Running Tukey HSD...")
            
            # Perform Tukey HSD to find pairwise differences
            tukey = pairwise_tukeyhsd(endog=df[var], groups=df['Course'], alpha=0.05)
            
            # Filter results: Save only pairs where the difference is significant (reject=True)
            tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
            sig_pairs = tukey_df[tukey_df['reject'] == True]
            
            # Save results to specific CSV for the metric
            sig_pairs.to_csv(f"reports/tables/tukey_{var}.csv", index=False)