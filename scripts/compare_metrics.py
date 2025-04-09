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

def give_recommendation(metrics_df, configs_df=None):
    """Provide recommendations of best model based on the metrics and configurations."""
    recommendations = {}
    for exp in metrics_df.index:
        # Added error handling for missing metrics
        if 'accuracy' in metrics_df.columns and not pd.isna(metrics_df.loc[exp, 'accuracy']):
            acc = metrics_df.loc[exp, 'accuracy']
            if acc > 0.9:
                recommendations[exp] = f"Good performance (accuracy: {acc:.4f})"
            elif acc > 0.7:
                recommendations[exp] = f"Average performance (accuracy: {acc:.4f})"
            else:
                recommendations[exp] = f"Poor performance (accuracy: {acc:.4f})"
        # Added fallback for other metrics
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
                recommendations[exp] = "No metrics available for evaluation"
    
    return recommendations

def main():
    parser = argparse.ArgumentParser(description="Compare experiment metrics and configurations.")
    parser.add_argument("--metrics_dir", type=str, default="experiments", 
                        help="Directory containing experiment metrics JSON files.")
    parser.add_argument("--configs_dir", type=str, default="experiments", 
                        help="Directory containing experiment configurations JSON files.")
    parser.add_argument("--save_path", type=str, default=None, 
                        help="Path to save the plot.")
    args = parser.parse_args()

    # Load data
    metrics = load_exp_metrics(args.metrics_dir)
    configs = load_exp_configs(args.configs_dir)

    if not metrics:
        print("Erorr 404 - No experiment metrics found")
        return

    # Convert to DataFrames
    metrics_df = conversion_to_df(metrics)
    configs_df = conversion_to_df(configs) if configs else None

    # Display metrics
    print("\nMetrics Comparison:")
    print(metrics_df)
    print("\n")

    # Plot metrics
    plot_metrics(metrics_df, args.save_path)
    
    # Generate recommendations
    recommendations = give_recommendation(metrics_df, configs_df)
    print("\nRecommendations:")
    for exp, rec in recommendations.items():
        print(f"- {exp}: {rec}")

if __name__ == "__main__":
    main()