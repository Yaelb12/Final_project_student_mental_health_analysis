import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import logging

def check_efa_assumptions(df, variables, logger):
    """
    Checks if the data is suitable for Factor Analysis using KMO and Bartlett's Test.
    Saves results to a CSV table.
    """
    logger.info("--- CHECKING EFA ASSUMPTIONS ---")
    subset = df[variables]
    
    # Bartlett's test: p-value should be < 0.05
    chi_square, p_value = calculate_bartlett_sphericity(subset)
    # KMO test: score should be > 0.6
    kmo_all, kmo_model = calculate_kmo(subset)
    
    results = {
        'Test': ['Bartlett Chi-Square', 'Bartlett P-Value', 'KMO Overall Score'],
        'Value': [round(chi_square, 3), round(p_value, 4), round(kmo_model, 3)],
        'Passed': [p_value < 0.05, p_value < 0.05, kmo_model > 0.6]
    }
    
    res_df = pd.DataFrame(results)
    res_df.to_csv("reports/tables/efa_assumptions.csv", index=False)
    logger.info(f"Assumptions results saved. KMO Score: {round(kmo_model, 3)}")
    return kmo_model > 0.6 and p_value < 0.05

def perform_efa(df, variables, logger, n_factors=1):
    """
    Performs Exploratory Factor Analysis to identify underlying latent factors.
    Returns the factor loadings.
    """
    logger.info(f"--- PERFORMING EFA (Factors={n_factors}) ---")
    subset = df[variables]
    
    # Initialize and fit Factor Analyzer
    fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
    fa.fit(subset)
    
    # Create loadings table (how much each variable contributes to the factor)
    loadings = pd.DataFrame(
        fa.loadings_, 
        index=variables, 
        columns=[f'Factor_{i+1}' for i in range(n_factors)]
    )
    
    loadings.to_csv("reports/tables/efa_loadings.csv")
    logger.info("EFA loadings saved to reports/tables/efa_loadings.csv")
    return loadings