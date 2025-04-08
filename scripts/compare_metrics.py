import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def load_exp_metrics(directory="experiments"):
    """Load experiment metrics from JSON files in the specified directory."""
    metrics = {}
    for file in Path(directory).glob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)
            metrics[file.stem] = data
    return metrics

def load_exp_configs(directory="experiments"):
    """Load experiment configurations from JSON files in the specified directory."""
    configs = {}
    for file in Path(directory).glob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)
            configs[file.stem] = data
    return configs

def conversion_to_df(metrics):
    return pd.DataFrame.from_dict(metrics, orient='index')

def plot_metrics(metrics_df,save_path=None):
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
    return fig

def give_recommendation(metrics_df, configs_df):
    """Provide recommendations based on the metrics and configurations."""
    recommendations = {}
    for exp in metrics_df.index:
        if metrics_df.loc[exp, 'accuracy'] > 0.9:
            recommendations[exp] = "Good performance"
        elif metrics_df.loc[exp, 'accuracy'] > 0.7:
            recommendations[exp] = "Average performance"
        else:
            recommendations[exp] = "Poor performance"
    return recommendations

def main():
    parser = argparse.ArgumentParser(description="Compare experiment metrics and configurations.")
    parser.add_argument("--metrics_dir", type=str, default="experiments", help="Directory containing experiment metrics JSON files.")
    parser.add_argument("--configs_dir", type=str, default="experiments", help="Directory containing experiment configurations JSON files.")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the plot.")
    args = parser.parse_args()

    metrics = load_exp_metrics(args.metrics_dir)
    configs = load_exp_configs(args.configs_dir)

    metrics_df = conversion_to_df(metrics)
    configs_df = conversion_to_df(configs)

    fig = plot_metrics(metrics_df, args.save_path)
    
    recommendations = give_recommendation(metrics_df, configs_df)
    for exp, rec in recommendations.items():
        print(f"Experiment: {exp}, Recommendation: {rec}")
    
