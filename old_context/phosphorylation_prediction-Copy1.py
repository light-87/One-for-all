#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory-Efficient Phosphorylation Site Prediction with XGBoost and Datatable

This script processes protein sequence data to predict phosphorylation sites,
using datatable for efficient data handling and XGBoost for prediction.
It includes comprehensive logging and performance visualization.
"""

import os
import gc
import time
import json
import argparse
import logging
import datetime
import numpy as np
import datatable as dt
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score
)
from tqdm import tqdm
from contextlib import contextmanager


# Configure logging
def setup_logging(log_dir="logs"):
    """Set up logging configuration"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"phosphorylation_xgb_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


@contextmanager
def timer(name):
    """Context manager for timing code blocks"""
    t0 = time.time()
    yield
    logger.info(f'{name} - done in {time.time() - t0:.2f}s')


def save_model_metrics(metrics, output_dir):
    """Save model metrics to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert NumPy arrays to lists for JSON serialization
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics[key] = value.tolist()
    
    with open(os.path.join(output_dir, f"metrics_{timestamp}.json"), 'w') as f:
        json.dump(metrics, f, indent=4)


def plot_training_metrics(evals_result, output_dir):
    """Plot training and validation metrics"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot Loss and AUC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Log Loss plot
    ax1.plot(evals_result['train']['logloss'], label='Train')
    ax1.plot(evals_result['validation']['logloss'], label='Validation')
    ax1.set_title('Log Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Log Loss')
    ax1.legend()
    ax1.grid(True)
    
    # AUC plot
    ax2.plot(evals_result['train']['auc'], label='Train')
    ax2.plot(evals_result['validation']['auc'], label='Validation')
    ax2.set_title('AUC')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('AUC')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_metrics_{timestamp}.png"), dpi=300)
    plt.close()


def plot_feature_importance(importance_dict, top_n=30, output_dir='plots'):
    """Plot top N most important features"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sort by importance
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
    
    # Take top N features
    top_features = list(sorted_importance.keys())[:top_n]
    importance_values = [sorted_importance[feature] for feature in top_features]
    
    # Plot
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(top_features)), importance_values, align='center')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Important Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"feature_importance_{timestamp}.png"), dpi=300)
    plt.close()
    
    return sorted_importance


def plot_model_performance(y_test, y_pred, y_pred_proba, output_dir='plots'):
    """Create performance plots for the model"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create a 2x2 grid for the plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    
    # ROC Curve plot
    ax1.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc="lower right")
    ax1.grid(True)
    
    # Precision-Recall Curve plot
    ax2.plot(recall, precision, lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True)
    
    # Confusion Matrix plot
    im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax3.set_title('Confusion Matrix')
    plt.colorbar(im, ax=ax3)
    tick_marks = [0, 1]
    ax3.set_xticks(tick_marks)
    ax3.set_yticks(tick_marks)
    ax3.set_xticklabels(['Negative', 'Positive'])
    ax3.set_yticklabels(['Negative', 'Positive'])
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
                    
    ax3.set_ylabel('True label')
    ax3.set_xlabel('Predicted label')
    
    # Bar chart of metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc
    }
    
    ax4.bar(metrics.keys(), metrics.values())
    ax4.set_title('Model Performance Metrics')
    ax4.set_ylim([0, 1.05])
    ax4.set_ylabel('Score')
    
    # Add values on top of bars
    for i, (key, value) in enumerate(metrics.items()):
        ax4.text(i, value + 0.02, f'{value:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"model_performance_{timestamp}.png"), dpi=300)
    plt.close()


def load_and_prepare_data(train_file, val_file, test_file):
    """Load training, validation, and testing data"""
    logger.info("Loading and preparing data...")
    
    with timer("Loading training data"):
        train_data = dt.fread(train_file)
        # Get all column names and filter out the ones we don't want
        all_cols = train_data.names
        feature_cols = [col for col in all_cols if col not in ["Header", "Position", "target"]]
        X_train = train_data[:, feature_cols]
        y_train = train_data[:, dt.f.target].to_numpy().flatten()
    
    with timer("Loading validation data"):
        val_data = dt.fread(val_file)
        # Use same feature columns from training data
        X_val = val_data[:, feature_cols]
        y_val = val_data[:, dt.f.target].to_numpy().flatten()
    
    with timer("Loading test data"):
        test_data = dt.fread(test_file)
        X_test = test_data[:, feature_cols]
        y_test = test_data[:, dt.f.target].to_numpy().flatten()
    
    # Clean up to free memory
    del train_data, val_data, test_data
    gc.collect()
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def convert_to_dmatrix(X_train, y_train, X_val, y_val, X_test, y_test):
    """Convert data to DMatrix format for XGBoost"""
    logger.info("Converting data to DMatrix format...")
    
    with timer("Converting training data to DMatrix"):
        # Convert datatable Frame to numpy array for XGBoost
        X_train_np = X_train.to_numpy()
        dtrain = xgb.DMatrix(X_train_np, label=y_train)
        # Free memory
        del X_train_np
        gc.collect()
    
    with timer("Converting validation data to DMatrix"):
        X_val_np = X_val.to_numpy()
        dval = xgb.DMatrix(X_val_np, label=y_val)
        del X_val_np
        gc.collect()
    
    with timer("Converting test data to DMatrix"):
        X_test_np = X_test.to_numpy()
        dtest = xgb.DMatrix(X_test_np, label=y_test)
        del X_test_np
        gc.collect()
    
    return dtrain, dval, dtest


def check_class_distribution(dtrain, dval, dtest):
    """Check and log the class distribution in datasets"""
    logger.info("Checking class distribution...")
    
    # Get labels
    train_labels = dtrain.get_label()
    val_labels = dval.get_label()
    test_labels = dtest.get_label()
    
    # Compute class distributions
    train_dist = np.unique(train_labels, return_counts=True)
    val_dist = np.unique(val_labels, return_counts=True)
    test_dist = np.unique(test_labels, return_counts=True)
    
    # Log distributions
    logger.info(f"Training set target distribution: {dict(zip(train_dist[0], train_dist[1]))}")
    logger.info(f"Validation set target distribution: {dict(zip(val_dist[0], val_dist[1]))}")
    logger.info(f"Test set target distribution: {dict(zip(test_dist[0], test_dist[1]))}")


def train_xgboost_model(dtrain, dval, params, num_boost_rounds=1000, early_stopping_rounds=50):
    """Train the XGBoost model"""
    logger.info("Training XGBoost model...")
    
    # Initialize dictionary to store evaluation results
    evals_result = {}
    
    with timer("XGBoost training"):
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_rounds,
            evals=[(dtrain, 'train'), (dval, 'validation')],
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=50
        )
    
    return model, evals_result


def evaluate_model(model, dtest, y_test):
    """Evaluate the model on test data"""
    logger.info("Evaluating model on test data...")
    
    with timer("Model evaluation"):
        # Make predictions
        y_pred_proba = model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Log results
        logger.info(f"Test Metrics:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        
        # Create metrics dictionary
        metrics = {
            "Accuracy": float(accuracy),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1),
            "ROC_AUC": float(roc_auc),
            "Confusion_Matrix": conf_matrix.tolist()
        }
    
    return metrics, y_pred, y_pred_proba


def analyze_feature_importance(model, output_dir="results"):
    """Analyze and log feature importance"""
    logger.info("Analyzing feature importance...")
    
    with timer("Feature importance analysis"):
        # Get feature importances based on gain
        importance_dict = model.get_score(importance_type='gain')
        
        # Sort by importance (descending)
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        
        # Log top 20 important features
        logger.info("Top 20 important features:")
        for i, (feature, score) in enumerate(list(sorted_importance.items())[:20]):
            logger.info(f"{i+1}. {feature}: {score:.4f}")
        
        # Save full feature importance to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, f"feature_importance_{timestamp}.json"), 'w') as f:
            json.dump(sorted_importance, f, indent=4)
    
    return sorted_importance


def main(args):
    """Main function to run the phosphorylation prediction pipeline"""
    global logger
    logger = setup_logging()
    logger.info("Starting phosphorylation site prediction with XGBoost")
    
    # Import pandas at the beginning to ensure it's available throughout the script
    import pandas as pd
    
    # Record start time
    start_time = time.time()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    
    try:
        # Load and prepare data
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_and_prepare_data(
            args.train_file, args.val_file, args.test_file
        )
        
        # Get feature names before conversion to DMatrix
        feature_names = X_train.names
        
        # Convert to DMatrix format
        dtrain, dval, dtest = convert_to_dmatrix(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Keep original test data for later use
        test_data_original = pd.read_csv(args.test_file)
        
        # Free memory for training data
        del X_train, X_val
        gc.collect()
        
        # Check class distribution
        check_class_distribution(dtrain, dval, dtest)
        
        # Set XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'eta': args.learning_rate,
            'max_depth': args.max_depth,
            'min_child_weight': args.min_child_weight,
            'subsample': args.subsample,
            'colsample_bytree': args.colsample_bytree,
            'tree_method': 'hist',
            'max_bin': 256
        }
        
        # Add GPU acceleration if requested
        if args.use_gpu:
            params['device'] = 'cuda'
        
        # Train model
        model, evals_result = train_xgboost_model(
            dtrain, dval, params, 
            num_boost_rounds=args.num_boost_rounds,
            early_stopping_rounds=args.early_stopping_rounds
        )
        
        # Plot training metrics
        plot_training_metrics(evals_result, args.plot_dir)
        
        # Save model
        model_path = os.path.join(args.output_dir, "phosphorylation_xgb_model.json")
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Evaluate model
        metrics, y_pred, y_pred_proba = evaluate_model(model, dtest, y_test)
        
        # Save metrics
        save_model_metrics(metrics, args.output_dir)
        
        # Save test predictions to CSV
        predictions_path = os.path.join(args.output_dir, "test_predictions.csv")
        save_test_predictions(y_test, y_pred, y_pred_proba, predictions_path, test_data_original)
        logger.info(f"Test predictions saved to {predictions_path}")
        
        # Plot model performance
        plot_model_performance(y_test, y_pred, y_pred_proba, args.plot_dir)
        
        # Analyze feature importance
        importance_dict = analyze_feature_importance(model, args.output_dir)
        
        # Plot feature importance
        plot_feature_importance(importance_dict, args.top_n_features, args.plot_dir)
        
        # Calculate total runtime
        total_runtime = time.time() - start_time
        logger.info(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
        
        logger.info("Phosphorylation site prediction completed successfully")
        
    except Exception as e:
        logger.exception("An error occurred during execution:")
        raise


def save_test_predictions(y_true, y_pred, y_pred_proba, output_path, X_test_data=None):
    """
    Save test set predictions to a CSV file with header, position, sequence, target, prediction, and probability
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted binary labels
        y_pred_proba (numpy.ndarray): Predicted probabilities
        output_path (str): Path to save the predictions CSV
        X_test_data (pandas.DataFrame, optional): Original test data containing headers, positions, and sequences
    """
    import pandas as pd
    
    # Create DataFrame with predictions
    if X_test_data is not None:
        # If original test data is provided, include all required columns
        predictions_df = pd.DataFrame({
            'Header': X_test_data['Header'],
            'Position': X_test_data['Position'],
            'target': y_true,
            'prediction': y_pred,
            'probability': y_pred_proba
        })
    else:
        # Fallback if original data not available (should be avoided)
        logger.warning("Original test data not provided. Creating predictions CSV with limited columns.")
        predictions_df = pd.DataFrame({
            'target': y_true,
            'prediction': y_pred,
            'probability': y_pred_proba
        })
    
    # Save to CSV
    predictions_df.to_csv(output_path, index=False)
    
    logger.info(f"Saved {len(predictions_df)} test predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost Phosphorylation Site Prediction")
    
    # Data arguments
    parser.add_argument("--train_file", type=str, default="split_data/train_data.csv",
                        help="Path to training data CSV")
    parser.add_argument("--val_file", type=str, default="split_data/val_data.csv",
                        help="Path to validation data CSV")
    parser.add_argument("--test_file", type=str, default="split_data/test_data.csv",
                        help="Path to test data CSV")
    
    # Model hyperparameters
    parser.add_argument("--learning_rate", type=float, default=0.1,
                        help="Learning rate for XGBoost")
    parser.add_argument("--max_depth", type=int, default=6,
                        help="Maximum tree depth for XGBoost")
    parser.add_argument("--min_child_weight", type=int, default=1,
                        help="Minimum sum of instance weight needed in a child")
    parser.add_argument("--subsample", type=float, default=0.8,
                        help="Subsample ratio of the training instances")
    parser.add_argument("--colsample_bytree", type=float, default=0.8,
                        help="Subsample ratio of columns when constructing each tree")
    parser.add_argument("--num_boost_rounds", type=int, default=1000,
                        help="Maximum number of boosting rounds")
    parser.add_argument("--early_stopping_rounds", type=int, default=50,
                        help="Number of rounds with no improvement for early stopping")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save model and results")
    parser.add_argument("--plot_dir", type=str, default="plots",
                        help="Directory to save plots")
    parser.add_argument("--top_n_features", type=int, default=30,
                        help="Number of top features to plot")
    
    # Hardware arguments
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU acceleration if available")
    
    args = parser.parse_args()
    main(args)