import os
import sys
import json
import shutil
import argparse
import logging
import pickle
import datetime
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def evaluate_classification_model(model, test_x, test_y):
    try:
        #Predictions
        y_pred = model.predict(test_x)

        #Metrics
        metrics = {}
        metrics["accuracy"] = float(accuracy_score(test_y, y_pred))

        #Other Metrics
        try:
            metrics["f1_score"] = float(f1_score(test_y, y_pred, average='weighted'))
            metrics["precision"] = float(precision_score(test_y, y_pred, average='weighted'))
            metrics["recall"] = float(recall_score(test_y, y_pred, average='weighted'))
        except Exception as e:
            logger.warning(f"Could not calculate some metrics: {str(e)}")
            metrics["f1_score"] = None
            metrics["precision"] = None
            metrics["recall"] = None

        #Confusion Matrix
        cm = confusion_matrix(test_y, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["loss"] = 0.0

        return metrics
    #In case of error
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        return {"accuracy": 0.0, "loss": 0.0}

def get_model_info(model):
    model_info = {}

    #Name
    model_info['model_name'] = model.__class__.__name__

    #Parameters
    if hasattr(model, 'get_params'):
        try:
            model_info['parameters'] = model.get_params()
        except:
            model_info['parameters'] = {}
    else:
        model_info['parameters'] = {}

    return model_info

def get_next_exp_number(experiments_dir="experiments"):
    # Create directory if it doesn't exist
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)
        return 1

    #Get folders with the same name pattern, here start with exp and a number
    exp_nums = []
    for item in os.listdir(experiments_dir):
        if os.path.isdir(os.path.join(experiments_dir, item)) and item.startswith("exp"):
            try:
                num = int(item[3:])
                exp_nums.append(num)
            except:
                continue

    # Return the next experiment number number
    return 1 if not exp_nums else max(exp_nums) + 1

def package_results(model, test_x=None, test_y=None, dataset_name="unknown_dataset", output_dir="experiments"):
    #Create the directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #Experimentation Number
    exp_num = get_next_exp_number(output_dir)
    exp_folder = f"exp{exp_num}"
    exp_path = os.path.join(output_dir, exp_folder)

    #Create the associated folder
    os.makedirs(exp_path)

    #Saving the model.pkl
    model_path = os.path.join(exp_path, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to {model_path}")

    #Evaluate model and metrics
    metrics_path = os.path.join(exp_path, "metrics.json")

    if test_x is not None and test_y is not None:
        logger.info("Evaluating model on test data...")
        metrics = evaluate_classification_model(model, test_x, test_y)
    else:
        # No test data provided, use placeholder metrics
        logger.warning("No test data provided. Using placeholder metrics.")
        metrics = {"accuracy": 0.0, "loss": 0.0}

    #Timestamp
    metrics["timestamp"] = datetime.datetime.now().isoformat()

    # Save the model's metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    #Save model parameters
    config_path = os.path.join(exp_path, "config.json")
    model_info = get_model_info(model)

    config = {
        "model_name": model_info.get("model_name", "UnknownModel"),
        "parameters": model_info.get("parameters", {}),
        "dataset": dataset_name,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_path}")

    #Success message
    logger.info(f"Successfully packaged model in {exp_folder}")
    logger.info(f"config.json, metrics.json and model.pkl have been created")

    return exp_path 
