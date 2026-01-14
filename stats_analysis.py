'''import os
import logging
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pointbiserialr
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Logging configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

os.makedirs("reports/figures", exist_ok=True)
sns.set(style="whitegrid")


# Load data

def load_clean(path="clean_data.csv"):
    logger.info("Loading cleaned dataset from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows and %d columns", df.shape[0], df.shape[1])
    return df


# Descriptive statistics

def describe_by_group(df, group='Is_STEM', outcome='Anxiety_Score'):
    logger.info("Computing descriptive statistics by group: %s", group)
    if group not in df.columns or outcome not in df.columns:
        logger.warning("Columns %s or %s not found", group, outcome)
        return pd.DataFrame()
    desc = df.groupby(group)[outcome].agg(['mean', 'std', 'count'])
    desc.to_csv("reports/describe_by_group.csv")
    logger.info("Saved descriptive statistics to reports/describe_by_group.csv")
    return desc


# Point-biserial correlation

def point_biserial(df, bin_col='Is_STEM', cont_col='Anxiety_Score'):
    logger.info("Running point-biserial correlation between %s and %s",
                bin_col, cont_col)
    if bin_col not in df.columns or cont_col not in df.columns:
        logger.warning("Columns %s or %s not found", bin_col, cont_col)
        out = {'r': np.nan, 'p': np.nan, 'n': 0}
    else:
        sub = df[[bin_col, cont_col]].dropna()
        if len(sub) < 3:
            logger.warning("Not enough data for correlation (n=%d)", len(sub))
            out = {'r': np.nan, 'p': np.nan, 'n': len(sub)}
        else:
            r, p = pointbiserialr(sub[bin_col], sub[cont_col])
            out = {'r': float(r), 'p': float(p), 'n': len(sub)}
    pd.DataFrame([out]).to_csv("reports/pointbiserial.csv", index=False)
    logger.info("Saved point-biserial results")
    return out


# Cohen's d

def cohens_d(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled = np.sqrt(((len(x) - 1) * x.var(ddof=1) +
                      (len(y) - 1) * y.var(ddof=1)) /
                     (len(x) + len(y) - 2))
    return (x.mean() - y.mean()) / pooled if pooled > 0 else np.nan


# Welch t-test

def welch_test(df, group_col='Is_STEM', outcome='Anxiety_Score'):
    logger.info("Running Welch t-test on %s by %s", outcome, group_col)
    if group_col not in df.columns or outcome not in df.columns:
        logger.warning("Columns %s or %s not found", group_col, outcome)
        res = {'t': np.nan, 'p': np.nan, 'cohens_d': np.nan}
    else:
        a = df[df[group_col] == 1][outcome].dropna()
        b = df[df[group_col] == 0][outcome].dropna()
        logger.info("Group sizes: %s=1 -> %d, %s=0 -> %d",
                    group_col, len(a), group_col, len(b))
        if len(a) < 2 or len(b) < 2:
            logger.warning("Not enough data for Welch test")
            res = {'t': np.nan, 'p': np.nan, 'cohens_d': np.nan}
        else:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            d = cohens_d(a, b)
            res = {'t': float(t), 'p': float(p), 'cohens_d': float(d)}
    pd.DataFrame([res]).to_csv("reports/welch_results.csv", index=False)
    logger.info("Saved Welch test results")
    return res


# Mann-Whitney U test

def mann_whitney(df, group_col='Is_STEM', outcome='Anxiety_Score'):
    logger.info("Running Mann-Whitney U test")
    if group_col not in df.columns or outcome not in df.columns:
        logger.warning("Columns %s or %s not found", group_col, outcome)
        out = {'U': np.nan, 'p': np.nan}
    else:
        a = df[df[group_col] == 1][outcome].dropna()
        b = df[df[group_col] == 0][outcome].dropna()
        if len(a) < 1 or len(b) < 1:
            logger.warning("Not enough data for Mann-Whitney")
            out = {'U': np.nan, 'p': np.nan}
        else:
            U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
            out = {'U': float(U), 'p': float(p)}
    pd.DataFrame([out]).to_csv("reports/mannwhitney.csv", index=False)
    logger.info("Saved Mann-Whitney results")
    return out


# Regression (OLS) with interaction

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

    # make sure predictors exist; if not, drop them from formula
    base_terms = ["Is_STEM", "SS_c"]
    extra_terms = []
    for col in ["Age", "Gender_num", "CGPA", "Semester_Credit_Load"]:
        if col in df.columns:
            extra_terms.append(col)
        else:
            logger.warning("Column %s not found, removed from model", col)
    rhs = "Is_STEM * SS_c"
    if extra_terms:
        rhs += " + " + " + ".join(extra_terms)
    formula = f"{outcome} ~ {rhs}"

    try:
        model = smf.ols(formula, data=df).fit(cov_type='HC3')
    except Exception as e:
        logger.error("Regression failed: %s", e)
        model = None

    if model is not None:
        try:
            coef = pd.read_html(
                model.summary().tables[1].as_html(),
                header=0, index_col=0
            )[0]
            coef.to_csv("reports/regression_summary.csv")
            logger.info("Saved regression summary to CSV")
        except Exception:
            logger.warning("Could not save regression summary table")

    logger.info("Regression completed")
    return model


# Regression diagnostics

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
    pd.DataFrame([out]).to_csv("reports/regression_diagnostics.csv",
                               index=False)
    logger.info("Saved regression diagnostics")
    return out


# Logistic regression

def logistic_check(df, outcome='Anxiety_Score', threshold=3):
    logger.info("Running logistic regression")
    if outcome not in df.columns:
        logger.warning("Outcome column %s not found; skipping logistic", outcome)
        return None

    df = df.copy()
    df['Anxiety_high'] = (df[outcome] >= threshold).astype(int)

    if 'Social_Support_num' not in df.columns:
        if 'Social_Support' in df.columns:
            df['Social_Support_num'] = pd.to_numeric(
                df['Social_Support'], errors='coerce'
            )
        else:
            df['Social_Support_num'] = np.nan

    if 'Gender_num' not in df.columns and 'Gender' in df.columns:
        df['Gender_num'] = pd.factorize(df['Gender'])[0]

    formula = "Anxiety_high ~ Is_STEM + Social_Support_num + Age + Gender_num + CGPA"
    # filter predictors that exist
    terms = []
    for term in ["Is_STEM", "Social_Support_num", "Age", "Gender_num", "CGPA"]:
        if term in df.columns:
            terms.append(term)
        else:
            logger.warning("Logistic: %s not in data, removed", term)
    if not terms:
        logger.warning("No predictors for logistic model; skipping")
        return None
    formula = "Anxiety_high ~ " + " + ".join(terms)

    try:
        logit = smf.logit(formula, data=df).fit(disp=False)
        try:
            coef = pd.read_html(
                logit.summary().tables[1].as_html(),
                header=0, index_col=0
            )[0]
            coef.to_csv("reports/logistic_summary.csv")
            logger.info("Saved logistic regression results")
        except Exception:
            logger.warning("Could not save logistic summary table")
        return logit
    except Exception as e:
        logger.warning("Logistic regression failed: %s", e)
        return None


# Plots

def plots(df):
    logger.info("Creating plots")

    # Boxplot Anxiety by Is_STEM
    if 'Is_STEM' in df.columns and 'Anxiety_Score' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='Is_STEM', y='Anxiety_Score', data=df)
        plt.title("Anxiety Score by Is_STEM")
        plt.savefig("reports/figures/box_anxiety.png", bbox_inches='tight')
        plt.close()
        logger.info("Saved box_anxiety.png")
    else:
        logger.warning("Cannot plot boxplot: columns missing")

    # Interaction: Social_Support_num x Is_STEM on Stress_Level
    if all(c in df.columns for c in ['Social_Support_num', 'Is_STEM', 'Stress_Level']):
        plt.figure(figsize=(7, 5))
        sns.pointplot(x='Social_Support_num', y='Stress_Level',
                      hue='Is_STEM', data=df, dodge=True, ci=95)
        plt.title("Interaction: Social Support x Is_STEM on Stress")
        plt.savefig("reports/figures/interaction_support_stress.png",
                    bbox_inches='tight')
        plt.close()
        logger.info("Saved interaction_support_stress.png")
    else:
        logger.warning("Cannot plot interaction: columns missing")

    # Correlation heatmap
    vars_list = [v for v in
                 ['Anxiety_Score', 'Stress_Level',
                  'Depression_Score', 'CGPA']
                 if v in df.columns]
    if len(vars_list) >= 2:
        sub = df[vars_list].dropna()
        if len(sub) > 0:
            corr = sub.corr()
            plt.figure(figsize=(6, 5))
            sns.heatmap(corr, annot=True, cmap='coolwarm',
                        vmin=-1, vmax=1)
            plt.title("Correlation heatmap")
            plt.savefig("reports/figures/corr.png", bbox_inches='tight')
            plt.close()
            logger.info("Saved corr.png")
        else:
            logger.warning("No data for correlation heatmap")
    else:
        logger.warning("Not enough numeric vars for heatmap")


# Run everything

def run_all(path="clean_data.csv"):
    df = load_clean(path)

    describe_by_group(df)
    point_biserial(df)
    welch_test(df)
    mann_whitney(df)

    model = run_regression(df)
    regression_diagnostics(model)

    logistic_check(df)
    plots(df)

    logger.info("All statistical analysis completed successfully")


if __name__ == "__main__":
    run_all()'''
