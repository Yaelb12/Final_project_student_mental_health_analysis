def plots(df):
    logger.info("Creating plots")

    # Boxplot
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Is_STEM', y='Anxiety_Score', data=df)
    plt.savefig("reports/figures/box_anxiety.png")
    plt.close()

    # Interaction plot
    if 'Social_Support_num' in df.columns:
        plt.figure(figsize=(7,5))
        sns.pointplot(x='Social_Support_num', y='Stress_Level', hue='Is_STEM', data=df)
        plt.savefig("reports/figures/interaction_support_stress.png")
        plt.close()

    # Correlation heatmap
    vars_list = ['Anxiety_Score','Stress_Level','Depression_Score','CGPA']
    sub = df[vars_list].dropna()
    corr = sub.corr()
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.savefig("reports/figures/corr.png")
    plt.close()

    logger.info("Plots saved")