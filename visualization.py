import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def annotate_significance(ax, variable, order):
    """
    Adds visual markers (red asterisks) above specific bars to indicate 
    statistical significance found in Tukey HSD post-hoc tests.
    """
    # Mapping each variable to the group that showed significant deviation in analysis
    highlights = {
        'Stress_Level': 'Medical',
        'Anxiety_Score': 'Law',
        'Depression_Score': 'Computer Science'
    }
    
    if variable in highlights:
        target_group = highlights[variable]
        # Iterate through the bars to find the match and place the star
        for i, bar_name in enumerate(order):
            if bar_name == target_group:
                # Get the height of the bar to position the text above it
                height = ax.patches[i].get_height()
                ax.text(i, height + 0.1, '*', ha='center', fontsize=25, 
                        color='red', fontweight='bold')

def create_bar_plot(df, variable, title, logger):
    """
    Generates a high-quality bar chart for a mental health metric.
    Includes Standard Error (SE) bars to represent data variability 
    and calls annotate_significance for post-hoc markers.
    """
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Sort courses by their mean values for a clearer, descending visualization
    order = df.groupby('Course')[variable].mean().sort_values(ascending=False).index
    
    # Draw the bar plot with standard error bars (SE)
    ax = sns.barplot(data=df, x='Course', y=variable, order=order, capsize=.1, errorbar='se')
    
    # Configure plot labels and titles
    plt.title(title, fontsize=15, pad=15)
    plt.ylabel(f"Average {variable.replace('_', ' ')}")
    plt.xlabel("Academic Course")
    plt.xticks(rotation=45)
    
    # Add post-hoc significance markers
    annotate_significance(ax, variable, order)
    
    # Save the chart as a high-resolution PNG
    plt.tight_layout()
    plt.savefig(f"reports/figures/{variable}_comparison.png")
    plt.close() # Free memory after saving
    logger.info(f"Scientific bar chart for {variable} saved successfully.")

def run_all_visualizations(df, logger):
    """
    The orchestrator function for standard supervised plots.
    Iterates through the primary mental health metrics and generates charts.
    """
    # Configuration for plot titles based on primary research findings
    plot_config = {
        'Stress_Level': 'Stress Levels Across Courses (Medical Group Significance)',
        'Anxiety_Score': 'Anxiety Scores Across Courses (Law Group Significance)',
        'Depression_Score': 'Depression Scores Across Courses (CS Group Significance)'
    }
    
    for var, title in plot_config.items():
        create_bar_plot(df, var, title, logger)

def plot_correlation_heatmap(df, variables, logger):
    """
    Generates a correlation heatmap to identify relationships between metrics.
    Essential for justifying the underlying structure before EFA.
    """
    plt.figure(figsize=(8, 6))
    # Calculate Pearson correlation coefficients
    corr_matrix = df[variables].corr()
    
    # Visualize the matrix using a heatmap with numerical annotations
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
    
    plt.title("Correlation Matrix: Mental Health Metrics", fontsize=14, pad=15)
    plt.tight_layout()
    
    # Save the heatmap for the factor analysis justification
    plt.savefig("reports/figures/correlation_heatmap.png")
    plt.close()
    logger.info("Correlation heatmap generated and saved in reports/figures/.")