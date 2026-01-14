# logistic.py
import logging
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)

def logistic_check(df, outcome='Anxiety_Score', threshold=3):
    logger.info("Running logistic regression")

    if outcome not in df.columns:
        logger.warning("Outcome %s not found; skipping logistic", outcome)
        return None

    df = df.copy()
    df['Anxiety_high'] = (df[outcome] >= threshold).astype(int)

    # Social support numeric
    if 'Social_Support_num' not in df.columns:
        if 'Social_Support' in df.columns:
            df['Social_Support_num'] = pd.to_numeric(
                df['Social_Support'], errors='coerce'
            )
        else:
            df['Social_Support_num'] = np.nan

    # Gender numeric
    if 'Gender_num' not in df.columns and 'Gender' in df.columns:
        df['Gender_num'] = pd.factorize(df['Gender'])[0]

    # Build formula
    predictors = ["Is_STEM", "Social_Support_num", "Age", "Gender_num", "CGPA"]
    predictors = [p for p in predictors if p in df.columns]

    if not predictors:
        logger.warning("No predictors available for logistic regression")
        return None

    formula = "Anxiety_high ~ " + " + ".join(predictors)
    logger.info("Logistic formula: %s", formula)

    try:
        logit = smf.logit(formula, data=df).fit(disp=False)
    except Exception as e:
        logger.warning("Logistic regression failed: %s", e)
        return None

    # Save coefficients
    try:
        coef = logit.summary2().tables[1]
        coef.to_csv("reports/logistic_summary.csv")
        logger.info("Saved logistic regression summary")
    except Exception:
        logger.warning("Could not save logistic summary table")

    return logit
