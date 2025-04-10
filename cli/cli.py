import click
import pickle
import subprocess
import os
import pandas as pd
import json
import matplotlib.pyplot as plt 
from pathlib import Path

from scripts.package_results import package_results
from scripts.compare_metrics import run_compare_metrics
from scripts.run_dashboard import run_dashboard_ui

@click.group()
def cli():
    """MLOps Mini Platform CLI"""
    pass

# for test
@cli.command()
def hello():
    """Test if CLI works"""
    click.echo("Hello from the MLOps CLI!")

# Call function 1-result_package

''' 
Design idea:
To match common real-world usage, we support test data input in CSV format.

In most machine learning datasets, features and labels are in the same table.
So, we allow users to specify the label column name via a parameter.

After receiving the file, the system automatically splits the data into features (X) and labels (y),
which are used for model evaluation and result packaging.

Finally, the package_results function from scripts/package_results.py is called to complete the packaging.
'''

@cli.command()
@click.option('--model-path', required=True, help='Path to model.pkl')
@click.option('--test-csv', default=None, help='Path to combined test CSV (features + label)')
@click.option('--label-col', default="label", help='Column name of label in CSV')
@click.option('--dataset-name', default="unknown_dataset", help='Name of dataset')
def package_results_cli(model_path, test_csv, label_col, dataset_name):
    """Package experiment results"""
    click.echo("Packaging experiment results ...")

    try:
        # Load model object from .pkl file
        # Deserialize the provided model.pkl file into a Python model object
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Prepare test data if CSV is provided
        # Split the test data into X (features) and y (labels) as numpy arrays
        test_x = None
        test_y = None

        if test_csv:
            # Load CSV file as a DataFrame
            df = pd.read_csv(test_csv)

            # Check if the label column exists to avoid user input errors
            if label_col not in df.columns:
                raise ValueError(f"Label column '{label_col}' not found in CSV.")

            # Split label and feature columns
            test_y = df[label_col].to_numpy()
            test_x = df.drop(columns=[label_col])

            click.echo(f"Loaded test CSV and split into features (X) and label column '{label_col}' (Y)")
        else:
            # If no CSV is provided, use default evaluation (accuracy and loss = 0.0)
            click.echo("No test CSV provided. Using default metrics.")

        # Call the core function package_results to generate experiment folder
        # Save model file, evaluation results, and model parameters into experiments/expN/
        output_path = package_results(model, test_x, test_y, dataset_name)
        click.echo(f"\nResults saved to: {output_path}")

    except Exception as e:
        # Catch and report any potential errors (e.g. wrong paths or bad formats)
        click.echo(f"Failed to package results: {str(e)}")


# Call function 2-compare_metrics
'''
Design idea:
This CLI command calls the run_compare_metrics function from scripts/compare_metrics.py.

The function loads metrics and configs, creates a bar chart, and gives model recommendations.

The CLI accepts folder paths, an optional save path, and a priority metric, then passes them to the function.
'''

@cli.command()
@click.option('--metrics-dir', default='experiments', help='Directory with experiment metrics JSON files')
@click.option('--configs-dir', default='experiments', help='Directory with experiment configs JSON files')
@click.option('--save-path', default=None, help='Path to save the comparison plot (e.g. comparison.png)')
@click.option('--priority-metric', 'priority_metric', default='accuracy', help='Priority metric for recommendations (e.g. accuracy, f1_score)')
def compare_metrics(metrics_dir, configs_dir, save_path, priority_metric):
    """
    Compare experiment metrics and provide recommendations.
    """
    click.echo("Running experiment metrics comparison...")
    run_compare_metrics(metrics_dir, configs_dir, save_path, priority_metric)

# Call function 3-run dashboard
'''
Design idea:
This CLI command calls run_dashboard_ui from scripts/run_dashboard.py.
Users can view experiment results and recommendations in their browser.
'''
@cli.command()
def run_dashboard():
    """Launch the Streamlit experiment comparison dashboard."""
    click.echo("Launching dashboard in background...")
    try:
        run_dashboard_ui()
        click.echo("Dashboard launched. Open your browser to view it.")
    except Exception as e:
        click.echo(f"Failed to launch dashboard: {str(e)}")


if __name__ == '__main__':
    cli()
