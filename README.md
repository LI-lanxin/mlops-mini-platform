# mlops-mini-platform
MLOps course project



---
# 🧠 MLOps Mini Platform

A lightweight command-line tool to help machine learning practitioners **package**, **compare**, and **visualize** their experiment results in a reproducible way.

This tool was developed for educational purposes, with a focus on simple, modular design and extensibility.

---

## 📁 Project Structure

```
mlops-mini-platform/
├── cli/                     # CLI entry point with main commands
│   ├── __init__.py
│   └── cli.py
├── scripts/                 # Core logic 
│   ├── __init__.py
│   ├── package_results.py
│   ├── compare_metrics.py
│   └── run_dashboard.py
├── experiments/             # Stores outputs like config.json, metrics.json, model.pkl
│   └── .gitkeep
├── utils/                   # Optional helper functions
│   └── helpers.py
├── tests/                   # Unit tests folder
├── pyproject.toml           # Configuration file 
└── README.md
```

---
🧪 The `experiments/` folder is used to store your experiment outputs (model, config, metrics).
This folder is not tracked in version control (see `.gitignore`) and will be empty after cloning the project.

Your own experiment results will be saved automatically when using the CLI or function interface.


---
## ⚙️ Main Features

| Feature               | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| 🧱 Package Results     | Save a trained model, its evaluation metrics, and config to a versioned folder |
| 📊 Compare Metrics     | Compare multiple experiments visually and get basic performance recommendation |
| 📈 Launch Dashboard   | Visualize experiments with an interactive Streamlit dashboard               |
| 🧪 Notebook Support   | Functions can be imported and used in Jupyter Notebooks                     |
| 💻 CLI Interface      | Use the `mlops` command to interact via terminal                            |

---
## 📦 Requirements

Before installing the project, make sure you have the following:

- Python 3.10+
- Git

Optional:

- VSCode or another IDE

---

## 🛠️ Installation 

### 🔁 Clone or Fork?

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

### 🔧 Set up the environment

```cmd
REM Step 1: Enter the project folder
cd mlops-mini-platform

REM Step 2: (Recommended) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate

REM Step 3: Install the tool in editable mode
pip install -e .
```

✅ This makes the `mlops` command available globally, and keeps your changes live.

---


---

## 🚀 How to Use (Command Line)

### 🔍 Show available commands

```cmd
mlops --help
```

### ✅ Run a test command

```cmd
mlops hello
```

Expected output:

```
Hello from the MLOps CLI!
```

### 📦 Package a model after training

> ✅ **Windows CMD users:** Please enter the entire command in one line:

```cmd
mlops package-results --model-path path/to/model.pkl --test-csv path/to/test.csv --label-col label --dataset-name my_dataset
```
加一个例子

Creates a new folder under `experiments/exp_n/` containing:
- `model.pkl`
- `metrics.json`
- `config.json`

### 📊 Compare experiment results

```cmd
mlops compare-metrics --metrics-dir experiments --configs-dir experiments --save-path comparison.png
```
加一个例子

Saves a bar plot comparing accuracy, F1, etc. and prints recommendations.

### 📈 Launch the Streamlit dashboard

```cmd
mlops dashboard
```

Opens a web interface in your browser to explore all experiments interactively.

---

## 📓 How to Use (Notebook)

You can also import the logic functions into any Python script or Jupyter Notebook:

```python
from scripts.package_results import package_results
from sklearn.ensemble import RandomForestClassifier

# Example: using the function manually
model = RandomForestClassifier().fit(X_train, y_train)
output_path = package_results(model, test_x=X_test, test_y=Y_test, dataset_name="demo")
print("Saved to:", output_path)
```

---

## 👥 Authors

- Alexandre LISSARDY
- Lanxin LI 
- Meng XIA
- Jiejie XU
- Bowei Zhao

---





## 🚀 3 Ways to Use This Tool

### 1. 📓 Use in Python/Notebook

```python
from scripts.package_results import package_results

package_results(model, test_x, test_y, dataset_name="my_model")
```

### 2. 🧪 Use as CLI Tool (with pip)

# Step 1: Install dependencies

pip install -r requirements.txt
# Step 2: Use CLI
python cli.py package-results-cli --model-path model.pkl --test-csv test.csv --label-col label --dataset-name demo

### 3. 🐳 Use with Docker (no Python needed)
# Step 1: Build Docker image
docker build -t mlops-cli-tool .

# Step 2: Run CLI
docker run -v $(pwd):/app mlops-cli-tool package-results-cli --model-path model.pkl --test-csv test.csv --label-col label --dataset-name demo
