# mlops-mini-platform
MLOps course project


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
