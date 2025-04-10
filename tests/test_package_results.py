from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from scripts.package_results import evaluate_classification_model, get_model_info, package_results

# test the evaluate_classification_model function, using the iris dataset from sklearn
def test_evaluate_model():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier().fit(X, y)
    metrics = evaluate_classification_model(model, X, y)
    assert "accuracy" in metrics
    assert metrics["accuracy"] > 0.8

# test the get_model_info function
def test_get_model_info():
    model = RandomForestClassifier(n_estimators=50)
    info = get_model_info(model)
    assert info["model_name"] == "RandomForestClassifier"
    assert info["parameters"]["n_estimators"] == 50

# test the package_results function, check if the output directory is created and if the files are created
def test_package_results(tmp_path):
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier().fit(X, y)
    out_dir = tmp_path / "experiments"
    
    exp_path = Path(package_results(model, X, y, output_dir=out_dir))

    assert (exp_path / "metrics.json").exists()
    assert (exp_path / "config.json").exists()
    assert (exp_path / "model.pkl").exists()
