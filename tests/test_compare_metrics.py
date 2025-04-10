import json
import pandas as pd
from scripts.compare_metrics import (
    load_exp_metrics,
    load_exp_configs,
    conversion_to_df,
    plot_metrics,
    give_recommendation
)

# test load_exp_metrics
def test_load_exp_metrics(tmp_path):
    exp = tmp_path / "exp1"
    exp.mkdir()
    (exp / "metrics.json").write_text(json.dumps({
        "accuracy": 0.85, "f1_score": 0.8, "precision": 0.82, "recall": 0.78
    }))
    result = load_exp_metrics(tmp_path)
    assert "exp1" in result
    assert result["exp1"]["accuracy"] == 0.85

# test load_exp_configs
def test_load_exp_configs(tmp_path):
    exp = tmp_path / "exp1"
    exp.mkdir()
    (exp / "config.json").write_text(json.dumps({
        "model_name": "RandomForest", "parameters": {"n_estimators": 100}
    }))
    result = load_exp_configs(tmp_path)
    assert "exp1" in result
    assert result["exp1"]["model_name"] == "RandomForest"

# test conversion_to_df, make sure it returns a dataframe
def test_conversion_to_df():
    data = {
        "exp1": {"accuracy": 0.9, "f1_score": 0.85},
        "exp2": {"accuracy": 0.8, "f1_score": 0.75}
    }
    df = conversion_to_df(data)
    assert isinstance(df, pd.DataFrame)
    assert "accuracy" in df.columns
    assert df.loc["exp1", "f1_score"] == 0.85

# test give_recommendation
def test_give_recommendation():
    df = pd.DataFrame({
        "accuracy": [0.95, 0.75, 0.6]
    }, index=["exp1", "exp2", "exp3"])
    recs = give_recommendation(df)
    assert recs["exp1"].startswith("Good")
    assert recs["exp2"].startswith("Average")
    assert recs["exp3"].startswith("Poor")

# test plot_metrics, make sure it returns a figure
def test_plot_metrics(tmp_path):
    df = pd.DataFrame({
        "accuracy": [0.85, 0.75],
        "f1_score": [0.8, 0.7]
    }, index=["exp1", "exp2"])
    save_path = tmp_path / "plot.png"
    fig = plot_metrics(df, save_path)
    assert save_path.exists()
    assert fig is not None
