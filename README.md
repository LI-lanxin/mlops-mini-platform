# mlops-mini-platform
MLOps course project



---
## MLOps Mini Platform

A lightweight tool designed to help machine learning practitioners **package**, **compare**, and **visualize** their experiment results in a reproducible way for **binary classification tasks**.



## 📁 Project Structure

```
MLOPS-MINI-PLATFORM/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI workflow file
├── cli/
│   ├── __init__.py
│   └── cli.py                      # Main CLI commands for package, compare, dashboard
├── experiments/
│   └── .gitkeep                    # Placeholder for experiment results (exp1, exp2, ...)
├── scripts/
│   ├── __init__.py
│   ├── compare_metrics.py         # Logic for comparing experiment metrics
│   ├── package_results.py         # Logic for packaging model results
│   └── run_dashboard.py           # Streamlit dashboard app
├── tests/
│   ├── test_compare_metrics.py     # Unit test for compare_metrics
│   └── test_package_results.py     # Unit test for package_results
├── .gitignore
├── HeartDiseasePrediction.ipynb    # Example notebook
├── pyproject.toml                  # Project metadata and CLI entrypoint
├── README.md
└── requirements.txt               # Python dependencies

```

The `experiments/` folder is used to store your experiment outputs (model, config, metrics).
This folder is not tracked in version control (see `.gitignore`) and will be empty after cloning the project.

Your own experiment results will be saved automatically when using the CLI or functions in notebook.

---
## Main Features

- **Automatic Metrics Discovery**: Scans experiment directories to find and load metrics from JSON files
- **Interactive Visualization**: Presents metrics in both tabular and chart formats
- **Model Recommendation**: Suggests the best model based on your selected priority metric
- **User-Friendly Interface**: Built with Streamlit for an intuitive user experience

---
##  Requirements

Before installing the project, make sure you have the following:

- Python 3.10+
- Git

Optional:

- VSCode or another IDE
---

##  Installation 

###  Clone or Fork?

If you only want to **use** this tool, you can simply clone this repository:

```bash
git clone https://github.com/LI-lanxin/mlops-mini-platform.git
```

If you want to **develop, modify, or contribute**, please **fork** this repository first on GitHub,  
then clone **your fork**:

```bash
git clone https://github.com/your-username/mlops-mini-platform.git
```

---

### Set up the environment

```cmd
Step 1: Enter the project folder
cd mlops-mini-platform

Step 2: (Recommended) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate

Step 3: Install the tool in editable mode
pip install -e .
```

This makes the `mlops` command available globally, and keeps your changes live.

**Note**: This tool provides the base environment for packaging and comparing models.
If your own notebooks require additional ML libraries (e.g., xgboost, lightgbm), you can install them manually within the same virtual environment,like: pip install xgboost lightgbm

---

## How to Use (Command Line)

### Show available commands

```cmd
mlops --help
```

### Run a test command

```cmd
mlops hello
```

Expected output:

```
Hello from the MLOps CLI!
```

### Package a model after training

To use this command, make sure you have already saved your trained model as a `.pkl` file (e.g., `model1.pkl`, `model2.pkl`, `model3.pkl`), along with a CSV file that includes both features and the label column. All files should be placed in the project root directory.

For example, to package models `model1`, `model2`, and `model3` using the same test dataset `test.csv`, with the label column named `label` and dataset name `test`, run the following commands:


```cmd
mlops package-results-cli --model-path model1.pkl --test-csv test.csv --label-col label --dataset-name test

mlops package-results-cli --model-path model2.pkl --test-csv test.csv --label-col label --dataset-name test

mlops package-results-cli --model-path model3.pkl --test-csv test.csv --label-col label --dataset-name test
```
Creates a new folder under `experiments/exp_n/` containing:
- `model.pkl`
- `metrics.json`
- `config.json`


### Compare experiment results

```cmd
mlops compare-metrics --metrics-dir experiments --configs-dir experiments --save-path comparison.png --priority-metric f1_score
```

**Note**: You can change `--priority-metric` to any available metric, such as `accuracy`, `recall`, or `precision`.

This command will generate a table comparing metrics across different models and recommend the best one based on your selected priority metric.  
It will also generate a bar chart (`comparison.png`) showing the performance comparison across models.

### Launch the Streamlit dashboard

```cmd
mlops run-dashboard
```

Opens a web interface in your browser to explore all experiments interactively.

---

## How to Use (Notebook)

You can also import the main functions into any Python script or Jupyter Notebook:

```python
from scripts.package_results import package_results
from scripts.compare_metrics import run_compare_metrics 
from scripts.run_dashboard import run_dashboard_ui

# Call our functions
package_results(model_recall, X_test, y_test, dataset_name="random_forest_heart")

metrics_df_acc, recommendations_acc, fig_acc = run_compare_metrics(
    metrics_dir="experiments",
    configs_dir="experiments",
    save_path=None,                   # Set to None to display the plot instead of saving
    priority_metric="accuracy"       # Set the priority metric (e.g., accuracy, f1_score)
)

run_dashboard_ui()
```

You can find a full example with outputs in the `HeartDiseasePrediction.ipynb` notebook included in this repository.

**Note:**  
If you are using these functions in a notebook, we recommend placing the notebook in the project root directory.  
Also, as mentioned in the *Set up the environment* section, running `pip install -e .` installs only the packages required to run this tool.  
If you want to train your own models in the same environment, please install the necessary additional packages manually.


##  Authors

- Alexandre LISSARDY
- Lanxin LI 
- Meng XIA
- Jiejie XU
- Bowei Zhao


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)