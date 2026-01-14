# regression.py
import logging
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def run_regression(df, outcome='Anxiety_Score'):
    logger.info("Running OLS regression with interaction")
    df = df.copy()

    # Social_Support_num
    if 'Social_Support_num' not in df.columns:
        if 'Social_Support' in df.columns:
            df['Social_Support_num'] = pd.to_numeric(
                df['Social_Support'], errors='coerce'
            )
            logger.info("Created Social_Support_num from Social_Support")
        else:
            logger.warning("No Social_Support column found; filling NaN")
            df['Social_Support_num'] = np.nan

    df['SS_c'] = df['Social_Support_num'] - df['Social_Support_num'].mean()

    # Gender_num
    if 'Gender_num' not in df.columns and 'Gender' in df.columns:
        df['Gender_num'] = pd.factorize(df['Gender'])[0]
        logger.info("Created Gender_num from Gender")

    # Build formula dynamically
    predictors = ["Is_STEM", "SS_c"]
    optional = ["Age", "Gender_num", "CGPA", "Semester_Credit_Load"]

    for col in optional:
        if col in df.columns:
            predictors.append(col)
        else:
            logger.warning("Column %s missing, removed from model", col)

    rhs = "Is_STEM * SS_c"
    extras = [col for col in predictors if col not in ["Is_STEM", "SS_c"]]
    if extras:
        rhs += " + " + " + ".join(extras)

    formula = f"{outcome} ~ {rhs}"
    logger.info("Regression formula: %s", formula)

    try:
        model = smf.ols(formula, data=df).fit(cov_type='HC3')
    except Exception as e:
        logger.error("Regression failed: %s", e)
        return None

    # Save coefficients
    try:
        coef = model.summary2().tables[1]
        coef.to_csv("reports/regression_summary.csv")
        logger.info("Saved regression summary")
    except Exception:
        logger.warning("Could not save regression summary table")

    return model


def regression_diagnostics(model):
    logger.info("Running regression diagnostics")
    out = {}

    if model is None:
        logger.warning("No model object; skipping diagnostics")
        out['breusch_pvalue'] = np.nan
    else:
        try:
            bp = sm.stats.diagnostic.het_breuschpagan(
                model.resid, model.model.exog
            )
            out['breusch_pvalue'] = float(bp[1])
        except Exception:
            out['breusch_pvalue'] = np.nan

    pd.DataFrame([out]).to_csv("reports/regression_diagnostics.csv", index=False)
    logger.info("Saved regression diagnostics")

    return out
