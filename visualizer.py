import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def annotate_significance(ax, variable, order):
    """
    Helper function to place significance stars above bars based on Tukey results.
    """
    # Map each variable to its most significant 'Outlier' group found in analysis
    highlights = {
        'Stress_Level': 'Medical',
        'Anxiety_Score': 'Law',
        'Depression_Score': 'Computer Science'
    }
    
    if variable in highlights:
        target_group = highlights[variable]
        for i, bar_name in enumerate(order):
            if bar_name == target_group:
                # Add an asterisk above the target bar
                height = ax.patches[i].get_height()
                ax.text(i, height + 0.1, '*', ha='center', fontsize=25, 
                        color='red', fontweight='bold')

def create_bar_plot(df, variable, title, logger):
    """
    Generates a high-quality bar chart for a metric, sorted by mean values.
    Includes Standard Error (SE) bars and significance markers.
    """
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Sort courses by mean to create a clear descending visualization
    order = df.groupby('Course')[variable].mean().sort_values(ascending=False).index
    
    # Build the bar plot with seaborn
    ax = sns.barplot(data=df, x='Course', y=variable, order=order, capsize=.1, errorbar='se')
    
    # Styling labels and titles
    plt.title(title, fontsize=15, pad=15)
    plt.ylabel(f"Average {variable.replace('_', ' ')}")
    plt.xticks(rotation=45)
    
    # Add significance star to the chart
    annotate_significance(ax, variable, order)
    
    # Save the figure to the output folder
    plt.tight_layout()
    plt.savefig(f"reports/figures/{variable}_comparison.png")
    plt.close() # Close figure to free up memory
    logger.info(f"Visualized chart for {variable} saved.")

def run_all_visualizations(df, logger):
    """
    Manager function to iterate through all metrics and create corresponding charts.
    """
    logger.info("--- STARTING DATA VISUALIZATION PROCESS ---")
    
    # Dictionary mapping variable names to their descriptive plot titles
    plot_config = {
        'Stress_Level': 'Average Stress Level by Course (Medical Significance)',
        'Anxiety_Score': 'Average Anxiety Score by Course (Law Significance)',
        'Depression_Score': 'Average Depression Score by Course (CS Significance)'
    }
    
    # Generate a plot for each metric in the configuration
    for var, title in plot_config.items():
        create_bar_plot(df, var, title, logger)



def plot_correlation_matrix(df, variables, logger):
    """
    Creates a visual Heatmap of the correlations between mental health metrics.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(8, 6))
    corr_matrix = df[variables].corr()
    
    # Generate the heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    
    plt.title("Correlation Matrix: Mental Health Metrics", fontsize=14)
    plt.tight_layout()
    plt.savefig("reports/figures/correlation_heatmap.png")
    plt.close()
    logger.info("Correlation heatmap saved to reports/figures/correlation_heatmap.png")