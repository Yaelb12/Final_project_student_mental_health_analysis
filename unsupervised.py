import pandas as pd
import logging
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

def run_unsupervised_analysis(df, variables, logger):
    """
    Manager function for unsupervised learning.
    It verifies assumptions (KMO/Bartlett) and then performs EFA to identify latent factors.
    """
    logger.info("--- STARTING UNSUPERVISED ANALYSIS PHASE ---")
    
    # Check if the data structure is suitable for factor analysis
    is_suitable = check_efa_assumptions(df, variables, logger)
    
    if is_suitable:
        # If assumptions pass, proceed to identifying the common mental health factor
        perform_efa(df, variables, logger)
    else:
        logger.warning("Data does not meet EFA requirements. Skipping Factor Analysis.")

def check_efa_assumptions(df, variables, logger):
    """
    Calculates KMO (sampling adequacy) and Bartlett's Test (sphericity).
    Saves the metrics to a CSV table to provide statistical justification for the EFA.
    """
    subset = df[variables]
    
    # Bartlett's Test: Evaluates if variables are related (p-value < 0.05 required)
    chi_square, p_value = calculate_bartlett_sphericity(subset)
    
    # KMO Test: Evaluates the proportion of variance among variables (score > 0.6 required)
    kmo_all, kmo_model = calculate_kmo(subset)
    
    # Consolidate results into a clear report for the user
    assumption_results = {
        'Statistical_Test': ['Bartlett Chi-Square', 'Bartlett P-Value', 'KMO Score'],
        'Result_Value': [round(chi_square, 3), round(p_value, 4), round(kmo_model, 3)],
        'Threshold_Requirement': ['N/A', 'p < 0.05', 'Score > 0.6']
    }
    
    # Save the decision-making table to the reports folder
    pd.DataFrame(assumption_results).to_csv("reports/tables/efa_assumptions.csv", index=False)
    logger.info(f"EFA assumptions calculated. KMO: {round(kmo_model, 3)}")
    
    return kmo_model > 0.6 and p_value < 0.05

def perform_efa(df, variables, logger):
    """
    Executes Exploratory Factor Analysis (EFA) to discover hidden patterns.
    It reduces Stress, Anxiety, and Depression into a single 'Common Distress' factor.
    """
    # Initialize the factor analyzer for 1 factor (Mental Distress)
    fa = FactorAnalyzer(n_factors=1, rotation="varimax")
    fa.fit(df[variables])
    
    # Extract 'Loadings' - showing how much each variable contributes to the hidden factor
    loadings_table = pd.DataFrame(
        fa.loadings_, 
        index=variables, 
        columns=['General_Distress_Factor']
    )
    
    # Save findings for interpretation in the research paper
    loadings_table.to_csv("reports/tables/efa_loadings.csv")
    logger.info("EFA successfully completed. Factor loadings saved to tables folder.")