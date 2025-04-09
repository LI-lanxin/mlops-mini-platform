import click
import pickle
import subprocess
import os
import pandas as pd
import json
import matplotlib.pyplot as plt 
from pathlib import Path

from scripts.package_results import package_results

@click.group()
def cli():
    """MLOps Mini Platform CLI"""
    pass

@cli.command()
def hello():
    """Test if CLI works"""
    click.echo("Hello from the MLOps CLI!")

# Call function 1-result_package
@cli.command()
@click.option('--model-path', required=True, help='Path to model.pkl')
@click.option('--test-csv', default=None, help='Path to combined test CSV (features + label)')
@click.option('--label-col', default="label", help='Column name of label in CSV')
@click.option('--dataset-name', default="unknown_dataset", help='Name of dataset')
def package_results_cli(model_path, test_csv, label_col, dataset_name):
    """Package experiment results"""
    click.echo("Packaging experiment results ...")

    try:
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load and split test CSV if provided
        test_x = None
        test_y = None

        if test_csv:
            df = pd.read_csv(test_csv)

            if label_col not in df.columns:
                raise ValueError(f"Label column '{label_col}' not found in CSV.")

            test_y = df[label_col].to_numpy()
            test_x = df.drop(columns=[label_col]).to_numpy()

            click.echo(f"Loaded test CSV and split into features (X) and label column '{label_col}' (Y)")
        else:
            click.echo("No test CSV provided. Using default metrics.")

        output_path = package_results(model, test_x, test_y, dataset_name)
        click.echo(f"\nResults saved to: {output_path}")

    except Exception as e:
        click.echo(f"Failed to package results: {str(e)}")

# Call function 2-compare_metrics
@cli.command()
@click.option('--metrics-dir', default="experiments", help="Directory with experiment metrics JSON files")
@click.option('--configs-dir', default="experiments", help="Directory with experiment configs JSON files")
@click.option('--save-path', default=None, help="Path to save the comparison plot (e.g. comparison.png)")
def compare_metrics(metrics_dir, configs_dir, save_path):
    """Compare experiment metrics and provide recommendations"""
    click.echo("Comparing experiment metrics...")

    def load_exp_json(directory):
        """Load all JSON files in a directory and return a dict."""
        data = {}
        for file in Path(directory).glob("*.json"):
            with open(file, "r") as f:
                content = json.load(f)
                data[file.stem] = content
        return data

    def conversion_to_df(d):
        return pd.DataFrame.from_dict(d, orient='index')

    def plot_metrics(metrics_df, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_df.plot(kind='bar', ax=ax)
        plt.title("Experiment Metrics Comparison")
        plt.xlabel("Experiment")
        plt.ylabel("Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            click.echo(f"Plot saved to: {save_path}")
        else:
            plt.show()
        return fig

    def give_recommendation(metrics_df):
        recommendations = {}
        for exp in metrics_df.index:
            acc = metrics_df.loc[exp].get('accuracy', 0)
            if acc > 0.9:
                recommendations[exp] = "Good performance"
            elif acc > 0.7:
                recommendations[exp] = "Average performance"
            else:
                recommendations[exp] = "Poor performance"
        return recommendations

    # Load data
    metrics = load_exp_json(metrics_dir)
    configs = load_exp_json(configs_dir)

    if not metrics:
        click.echo("No metrics found.")
        return

    metrics_df = conversion_to_df(metrics)
    configs_df = conversion_to_df(configs)

    # Plot metrics
    plot_metrics(metrics_df, save_path)

    # Recommendations
    recommendations = give_recommendation(metrics_df)
    click.echo("\nRecommendations:")
    for exp, rec in recommendations.items():
        click.echo(f"- {exp}: {rec}")

# Call function 3-run dashboard
@cli.command()
def run_dashboard():
    """Launch the Streamlit experiment comparison dashboard"""
    click.echo("Launching dashboard...")
    try:
        subprocess.run(["streamlit", "run", "scripts/run_dashboard.py"])
    except Exception as e:
        click.echo(f"Failed to launch dashboard: {str(e)}")

if __name__ == '__main__':
    cli()
