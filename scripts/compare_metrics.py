import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def load_exp_metrics(directory="experiments"):
    """Load experiment metrics from JSON files in the specified directory."""
    metrics = {}
    for path in Path(directory).glob("*/metrics.json"):
        with open(path) as f:
            data = json.load(f)
        metrics[path.parent.name] = data
    return metrics

def load_exp_configs(directory="experiments"):
    """Load experiment configurations from JSON files in the specified directory."""
    configs = {}
    for path in Path(directory).glob("*/config.json"):
        with open(path) as f:
            data = json.load(f)
        configs[path.parent.name] = data
    return configs

def conversion_to_df(metrics):
    """Convert metrics dictionary to pd DataFrame."""
    return pd.DataFrame.from_dict(metrics, orient='index')

def plot_metrics(metrics_df, save_path=None):
    """Plot the metrics from the DataFrame."""
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df.plot(kind='bar', ax=ax)
    plt.title("Experiment Metrics Comparison")
    plt.xlabel("Experiment")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Chart saved to {save_path}")
    return fig

def give_recommendation(metrics_df, configs_df=None, priority_metric='None'):
    """Provide recommendations of best model based on the metrics and configurations."""
    recommendations = {}
    
    # First see if there is a priority metric available
    best_model = None
    best_score = -1
    
    # First pass to find the best model according to priority metric
    for exp in metrics_df.index:
        if priority_metric in metrics_df.columns and not pd.isna(metrics_df.loc[exp, priority_metric]):
            score = metrics_df.loc[exp, priority_metric]
            if score > best_score:
                best_score = score
                best_model = exp
    
    # Second pass to generate recommendations for each model
    for exp in metrics_df.index:
        # Check if priority metric exists for this experiment
        if priority_metric in metrics_df.columns and not pd.isna(metrics_df.loc[exp, priority_metric]):
            val = metrics_df.loc[exp, priority_metric]
            is_best = ""
            if exp == best_model:
                is_best = " (BEST MODEL)"
            
            if val > 0.9:
                recommendations[exp] = f"Good performance ({priority_metric}: {val:.4f}){is_best}"
            elif val > 0.7:
                recommendations[exp] = f"Average performance ({priority_metric}: {val:.4f}){is_best}"
            else:
                recommendations[exp] = f"Poor performance ({priority_metric}: {val:.4f}){is_best}"
                
        # if no priority metric, first use accuracy, then F1-score
        elif 'accuracy' in metrics_df.columns and not pd.isna(metrics_df.loc[exp, 'accuracy']):
            acc = metrics_df.loc[exp, 'accuracy']
            if acc > 0.9:
                recommendations[exp] = f"Good performance (accuracy: {acc:.4f})"
            elif acc > 0.7:
                recommendations[exp] = f"Average performance (accuracy: {acc:.4f})"
            else:
                recommendations[exp] = f"Poor performance (accuracy: {acc:.4f})"
                
        elif 'f1_score' in metrics_df.columns and not pd.isna(metrics_df.loc[exp, 'f1_score']):
            f1 = metrics_df.loc[exp, 'f1_score']
            if f1 > 0.9:
                recommendations[exp] = f"Good performance (F1-score: {f1:.4f})"
            elif f1 > 0.7:
                recommendations[exp] = f"Average performance (F1-score: {f1:.4f})"
            else:
                recommendations[exp] = f"Poor performance (F1-score: {f1:.4f})"
                
        else:
            # Use any available metric
            for metric in metrics_df.columns:
                if not pd.isna(metrics_df.loc[exp, metric]):
                    val = metrics_df.loc[exp, metric]
                    recommendations[exp] = f"Performance based on {metric}: {val:.4f}"
                    break
            else:
                recommendations[exp] = "Error 404 No metrics available for evaluation"
    
    # Add summary recommendation about best model
    if best_model:
        best_value = metrics_df.loc[best_model, priority_metric]
        summary = f"RECOMMENDATION: Model '{best_model}' is the best performer with {priority_metric} = {best_value:.4f}"
        recommendations['summary'] = summary
    
    return recommendations


def run_compare_metrics(metrics_dir="experiments", configs_dir="experiments", save_path=None, priority_metric='accuracy'):
    """
    Run the metrics comparison and generate recommendations.
    
    Parameters:
    -----------
    metrics_dir : str
        Directory containing experiment metrics JSON files
    configs_dir : str
        Directory containing experiment configuration JSON files
    save_path : str or None
        Path to save the comparison plot, if not specified, the program will show the plot but not save it (default: None)
    priority_metric : str
        Metric to use when giving recommendation (default: 'accuracy')
        
    Returns:
    --------
    tuple
        (metrics_df, recommendations, plot_fig)
    """
    # Load data
    metrics = load_exp_metrics(metrics_dir)
    configs = load_exp_configs(configs_dir)
    
    if not metrics:
        print("Error 404 - No experiment metrics found")
        return None, None, None
    
    # Convert to DataFrames
    metrics_df = conversion_to_df(metrics)
    configs_df = conversion_to_df(configs) if configs else None
    
    # Display metrics
    print("\nMetrics Comparison:")
    print(metrics_df)
    print("\n")
    
    # Plot metrics
    fig = plot_metrics(metrics_df, save_path)
    
    # Generate recommendations with priority metric
    recommendations = give_recommendation(metrics_df, configs_df, priority_metric)
    print("\nRecommendations:")
    for exp, rec in recommendations.items():
        print(f"- {exp}: {rec}")
    
    return metrics_df, recommendations, fig

def main():
    parser = argparse.ArgumentParser(description="Compare experiment metrics and configurations.")
    parser.add_argument("--metrics_dir", type=str, default="experiments",
                        help="Directory containing experiment metrics JSON files.")
    parser.add_argument("--configs_dir", type=str, default="experiments",
                        help="Directory containing experiment configurations JSON files.")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save the plot.")
    parser.add_argument("--priority", type=str, default="accuracy",
                        help="Priority metric used for recommendations, can be defined directly or selected in streamlit interface")
    args = parser.parse_args()
    
    run_compare_metrics(args.metrics_dir, args.configs_dir, args.save_path, args.priority)

if __name__ == "__main__":
    main()