# -*- coding: utf-8 -*-
"""Run_dashboard- Meng XIA.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19YECx-f0C0I6_JXI3ZsvCk2rDqSAfEAb
"""

import streamlit as st  # For visualization
import pandas as pd  # For organizing experimental metrics data, need to display in tables
import matplotlib.pyplot as plt  # For creating bar charts
from pathlib import Path  # For cross-platform file path handling
import json  # Because the metric files we need to read are in JSON format
from compare_metrics import give_recommendation  # Because we need to call the recommendation model function from compare_metrics
import glob # for path finding
import os # Import the operating system module for path operation

# Build a function to load all metrics content in the experiment folder and save it as a dict
def load_all_metrics(exp_dir="experiments"):
    metrics = {}
    for path in glob.glob(os.path.join(Path(exp_dir), "*/metrics.json")):
        with open(path) as f:
            data = json.load(f)
        # Extract experiment name from path
        exp_name = os.path.basename(os.path.dirname(path))
        metrics[exp_name] = data
    return metrics

# Build a function to create plots
def plot_metrics(metrics_dict):
    """
    Plot all the metrics
    arguments: metrics_dict : dictionary
    return: None
    """
    # first transpose the previously constructed dictionary format metrics
    # to make the X-axis represent different experiments and the Y-axis represent values
    metrics_df = pd.DataFrame(metrics_dict).T
    # First display as a df table
    st.write("Experiment Metrics Data Table:")
    st.dataframe(metrics_df)
    #then use a bar chart to show comparison between different experiments
    st.write("🔎 Metrics Comparison Chart:")
    fig, ax = plt.subplots()
    metrics_df.plot(kind='bar', ax=ax)
    plt.title("Experiment Metrics Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45) # in case the name is too long
    st.pyplot(fig)

# Streamlit page configuration
st.set_page_config(page_title="Experiment Comparison Platform", layout="centered")
st.title("Experiment Comparison Platform")
st.markdown("The system will automatically compare multiple model experiments, their metrics, and provide recommendations.")

# Load the results of the first function load_metrics into a new variable
metrics = load_all_metrics()
# If these loaded metrics are empty
if len(metrics) == 0:
    st.warning("No experiments have been found, please upload experiment folders containing metrics.json first.")
else:
    # Visualization + Recommendation output
    plot_metrics(metrics)

    st.markdown("## System Recommendation")
    available_metrics = list(next(iter(metrics.values())).keys())
    # we just want to give these priority metric options
    metric_options = [m for m in ["accuracy", "f1_score", "precision", "recall", "loss"] if m in available_metrics]
    priority_metric = st.selectbox("Select priority metric", metric_options)

    # Convert metrics dictionary to DataFrame before calling give_recommendation
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    model_suggestion = give_recommendation(metrics_df, priority_metric=priority_metric)  
    st.success(model_suggestion)