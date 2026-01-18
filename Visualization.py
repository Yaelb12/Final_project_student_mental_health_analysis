# plots.py
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


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
        logger.warning("Missing columns for boxplot")

    # Interaction plot
    needed = ['Social_Support_num', 'Is_STEM', 'Stress_Level']
    if all(c in df.columns for c in needed):
        plt.figure(figsize=(7, 5))
        sns.pointplot(
            x='Social_Support_num',
            y='Stress_Level',
            hue='Is_STEM',
            data=df,
            dodge=True,
            ci=95
        )
        plt.title("Interaction: Social Support × Is_STEM on Stress")
        plt.savefig("reports/figures/interaction_support_stress.png",
                    bbox_inches='tight')
        plt.close()
        logger.info("Saved interaction_support_stress.png")
    else:
        logger.warning("Missing columns for interaction plot")

    # Correlation heatmap
    vars_list = [
        v for v in ['Anxiety_Score', 'Stress_Level', 'Depression_Score', 'CGPA']
        if v in df.columns
    ]

    if len(vars_list) >= 2:
        sub = df[vars_list].dropna()
        if len(sub) > 0:
            corr = sub.corr()
            plt.figure(figsize=(6, 5))
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title("Correlation Heatmap")
            plt.savefig("reports/figures/corr.png", bbox_inches='tight')
            plt.close()
            logger.info("Saved corr.png")
        else:
            logger.warning("No data for heatmap")
    else:
        logger.warning("Not enough numeric columns for heatmap")
