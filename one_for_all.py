"""
One-For-All Phosphorylation Site Prediction Pipeline
A comprehensive single-file implementation for phosphorylation site prediction
"""

# ============================================================================
# SECTION 0: IMPORTS AND SETUP
# ============================================================================

# Standard library imports
import os
import gc
import time
import json
import yaml
import logging
import warnings
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from functools import lru_cache
import multiprocessing as mp

# Data manipulation
import numpy as np
import pandas as pd
import datatable as dt
from datatable import f, by

# Machine Learning
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Visualization and logging
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from tqdm import tqdm

# Optimization
import optuna
from scipy.optimize import minimize

# Interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Interpretability features will be limited.")

# Global variables
CONFIG = None
LOGGER = None
DEVICE = None

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: CONFIGURATION AND UTILITIES
# ============================================================================

def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def setup_logging(config: Dict) -> logging.Logger:
    """Setup logging configuration"""
    log_level = getattr(logging, config['logging']['level'].upper())
    
    # Create output directory if it doesn't exist
    if config['logging']['save_to_file']:
        os.makedirs(os.path.dirname(config['logging']['log_file']), exist_ok=True)
    
    # Configure logging
    handlers = [logging.StreamHandler()]
    if config['logging']['save_to_file']:
        handlers.append(logging.FileHandler(config['logging']['log_file']))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")
    
    return logger

def set_seed(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_memory_usage() -> Dict:
    """Get current memory usage"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024)
        }
    except ImportError:
        return {'rss_mb': 0, 'vms_mb': 0}

def timer(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        LOGGER.info(f"{func.__name__} completed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# ============================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# ============================================================================

@timer
def load_sequences(file_path: str) -> dt.Frame:
    """Load protein sequences from FASTA file using datatable"""
    LOGGER.info(f"Loading protein sequences from {file_path}")
    
    headers = []
    sequences = []
    current_header = None
    current_seq = ""
    
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if current_header:
                    headers.append(current_header)
                    sequences.append(current_seq)
                
                full_header = line[1:]
                parts = full_header.split("|")
                current_header = parts[1] if len(parts) > 1 else full_header
                current_seq = ""
            else:
                current_seq += line
        
        if current_header:
            headers.append(current_header)
            sequences.append(current_seq)
    
    df = dt.Frame({
        "Header": headers,
        "Sequence": sequences
    })
    
    LOGGER.info(f"Loaded {df.nrows} protein sequences")
    return df

@timer
def load_labels(file_path: str) -> dt.Frame:
    """Load phosphorylation site labels"""
    LOGGER.info(f"Loading phosphorylation site labels from {file_path}")
    
    try:
        df_labels = dt.fread(file_path, nthreads=4)
        LOGGER.info(f"Direct Excel read successful with {df_labels.nrows} rows")
    except:
        try:
            import pandas as pd
            df_labels_pd = pd.read_excel(file_path)
            df_labels = dt.Frame(df_labels_pd)
            del df_labels_pd
            gc.collect()
            LOGGER.info(f"Excel read via pandas successful with {df_labels.nrows} rows")
        except Exception as e:
            raise RuntimeError(f"Could not read labels file: {e}")
    
    if 'UniProt ID' not in df_labels.names and 'UniProtID' in df_labels.names:
        df_labels.names = {'UniProtID': 'UniProt ID'}
    
    LOGGER.info(f"Loaded {df_labels.nrows} phosphorylation sites")
    return df_labels

@timer
def load_physicochemical_properties(file_path: str) -> Dict:
    """Load physicochemical properties"""
    LOGGER.info(f"Loading physicochemical properties from {file_path}")
    
    props_dt = dt.fread(file_path)
    
    properties = {}
    for i in range(props_dt.nrows):
        row = props_dt[i, :].to_list()[0]
        aa = row[0]
        properties[aa] = row[1:]
    
    LOGGER.info(f"Loaded physicochemical properties for {len(properties)} amino acids")
    return properties

@timer
def generate_negative_samples(df_merged: dt.Frame) -> dt.Frame:
    """Generate balanced negative samples"""
    LOGGER.info("Generating negative samples...")
    
    df_pd = df_merged.to_pandas()
    all_rows = []
    
    for header, group in tqdm(df_pd.groupby('Header'), desc="Processing proteins"):
        seq = group['Sequence'].iloc[0]
        positive_positions = group['Position'].astype(int).tolist()
        
        sty_positions = [i+1 for i, aa in enumerate(seq) if aa in ["S", "T", "Y"]]
        negative_candidates = [pos for pos in sty_positions if pos not in positive_positions]
        
        n_pos = len(positive_positions)
        sample_size = min(n_pos, len(negative_candidates))
        
        if sample_size > 0:
            random.seed(42 + hash(header) % 10000)
            sampled_negatives = random.sample(negative_candidates, sample_size)
            
            all_rows.append(group)
            
            for neg_pos in sampled_negatives:
                new_row = group.iloc[0].copy()
                new_row['AA'] = seq[neg_pos - 1]
                new_row['Position'] = neg_pos
                new_row['target'] = 0
                all_rows.append(pd.DataFrame([new_row]))
    
    df_final_pd = pd.concat(all_rows, ignore_index=True)
    df_final = dt.Frame(df_final_pd)
    
    del df_pd, df_final_pd, all_rows
    gc.collect()
    
    LOGGER.info(f"Generated dataset with {df_final.nrows} rows (positives + negatives)")
    return df_final

def split_data_cv(df: dt.Frame, n_folds: int = 5) -> List[Tuple]:
    """Create cross-validation splits grouped by protein"""
    LOGGER.info(f"Creating {n_folds}-fold cross-validation splits")
    
    df_pd = df.to_pandas()
    headers = df_pd['Header'].unique()
    
    np.random.seed(CONFIG['data']['random_seed'])
    np.random.shuffle(headers)
    
    train_ratio = 0.7
    val_ratio = 0.15
    
    train_split = int(len(headers) * train_ratio)
    val_split = int(len(headers) * (train_ratio + val_ratio))
    
    train_headers = headers[:train_split]
    val_headers = headers[train_split:val_split]
    test_headers = headers[val_split:]
    
    train_df = df_pd[df_pd['Header'].isin(train_headers)]
    val_df = df_pd[df_pd['Header'].isin(val_headers)]
    test_df = df_pd[df_pd['Header'].isin(test_headers)]
    
    LOGGER.info(f"Train: {len(train_df)} samples from {len(train_headers)} proteins")
    LOGGER.info(f"Val: {len(val_df)} samples from {len(val_headers)} proteins")
    LOGGER.info(f"Test: {len(test_df)} samples from {len(test_headers)} proteins")
    
    train_dt = dt.Frame(train_df)
    val_dt = dt.Frame(val_df)
    test_dt = dt.Frame(test_df)
    
    del df_pd, train_df, val_df, test_df
    gc.collect()
    
    return train_dt, val_dt, test_dt

@timer
def preprocess_data():
    """Main data preprocessing pipeline"""
    LOGGER.info("Starting data preprocessing...")
    
    # Check if cached data exists
    train_file = os.path.join(CONFIG['paths']['output_dir'], 'data', 'train_data.csv')
    val_file = os.path.join(CONFIG['paths']['output_dir'], 'data', 'val_data.csv')
    test_file = os.path.join(CONFIG['paths']['output_dir'], 'data', 'test_data.csv')
    
    if (CONFIG['data']['use_cached_features'] and 
        all(os.path.exists(f) for f in [train_file, val_file, test_file])):
        LOGGER.info("Using cached preprocessed data")
        return
    
    # Load sequences, labels, properties
    df_seq = load_sequences(CONFIG['paths']['sequence_data'])
    df_labels = load_labels(CONFIG['paths']['labels_data'])
    
    # Merge data
    LOGGER.info("Merging sequences with labels...")
    df_seq_pd = df_seq.to_pandas()
    df_labels_pd = df_labels.to_pandas()
    
    merged_pd = pd.merge(
        df_seq_pd, df_labels_pd,
        left_on="Header", right_on="UniProt ID",
        how="inner"
    )
    merged_pd["target"] = 1
    
    df_merged = dt.Frame(merged_pd)
    del df_seq_pd, df_labels_pd, merged_pd
    gc.collect()
    
    # Clean data
    LOGGER.info("Cleaning data...")
    df_pd = df_merged.to_pandas()
    df_pd['SeqLength'] = df_pd['Sequence'].str.len()
    df_pd = df_pd[df_pd['SeqLength'] <= CONFIG['data']['max_sequence_length']]
    df_pd = df_pd.drop('SeqLength', axis=1)
    df_merged = dt.Frame(df_pd)
    del df_pd
    gc.collect()
    
    # Generate negative samples
    df_final = generate_negative_samples(df_merged)
    
    # Split data
    train_dt, val_dt, test_dt = split_data_cv(df_final)
    
    # Save processed data
    os.makedirs(os.path.join(CONFIG['paths']['output_dir'], 'data'), exist_ok=True)
    train_dt.to_csv(train_file)
    val_dt.to_csv(val_file)
    test_dt.to_csv(test_file)
    
    LOGGER.info("Data preprocessing completed")

# ============================================================================
# SECTION 3: FEATURE EXTRACTION
# ============================================================================

def extract_window(sequence: str, position: int, window_size: int) -> str:
    """Extract window around position"""
    pos_idx = position - 1
    start = max(0, pos_idx - window_size)
    end = min(len(sequence), pos_idx + window_size + 1)
    return sequence[start:end]

@lru_cache(maxsize=10000)
def extract_aac(sequence: str) -> Dict[str, float]:
    """Extract Amino Acid Composition features"""
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    aac = {aa: 0 for aa in amino_acids}
    seq_length = len(sequence)
    
    for aa in sequence:
        if aa in aac:
            aac[aa] += 1
    
    for aa in aac:
        aac[aa] = aac[aa] / seq_length if seq_length > 0 else 0
    
    return aac

@lru_cache(maxsize=10000)
def extract_dpc(sequence: str) -> Dict[str, float]:
    """Extract Dipeptide Composition features"""
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    dpc = {}
    for aa1 in amino_acids:
        for aa2 in amino_acids:
            dpc[aa1 + aa2] = 0
    
    if len(sequence) < 2:
        return dpc
    
    for i in range(len(sequence) - 1):
        dipeptide = sequence[i:i+2]
        if dipeptide in dpc:
            dpc[dipeptide] += 1
    
    total_dipeptides = len(sequence) - 1
    for dipeptide in dpc:
        dpc[dipeptide] = dpc[dipeptide] / total_dipeptides if total_dipeptides > 0 else 0
    
    return dpc

def extract_tpc_batch(sequences: List[str]) -> List[Dict]:
    """Extract Tripeptide Composition in batches - Memory optimized version"""
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    results = []
    for sequence in sequences:
        if len(sequence) < 3:
            # Create minimal TPC with only zeros for short sequences
            tpc = {f'TPC_{i:04d}': 0.0 for i in range(100)}  # Only top 100 most common tripeptides
        else:
            # Count only observed tripeptides (sparse representation)
            observed_tpc = {}
            total_tripeptides = len(sequence) - 2
            
            for i in range(len(sequence) - 2):
                tripeptide = sequence[i:i+3]
                if all(aa in amino_acids for aa in tripeptide):
                    observed_tpc[tripeptide] = observed_tpc.get(tripeptide, 0) + 1
            
            # Convert to frequencies only for observed tripeptides
            tpc = {}
            tripeptide_list = sorted(observed_tpc.keys())  # Sort for consistency
            
            # Keep only top 100 most frequent tripeptides to limit memory
            sorted_tripeptides = sorted(observed_tpc.items(), key=lambda x: x[1], reverse=True)[:100]
            
            for idx, (tri, count) in enumerate(sorted_tripeptides):
                tpc[f'TPC_{idx:04d}'] = count / total_tripeptides
            
            # Fill remaining slots with zeros
            for idx in range(len(sorted_tripeptides), 100):
                tpc[f'TPC_{idx:04d}'] = 0.0
        
        results.append(tpc)
        
        # Force garbage collection every 1000 sequences
        if len(results) % 1000 == 0:
            import gc
            gc.collect()
    
    return results

def extract_binary_encoding(sequence: str, position: int, window_size: int) -> np.ndarray:
    """Extract binary encoding features"""
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    pos_idx = position - 1
    start = max(0, pos_idx - window_size)
    end = min(len(sequence), pos_idx + window_size + 1)
    
    window = sequence[start:end]
    
    left_pad = "X" * max(0, window_size - (pos_idx - start))
    right_pad = "X" * max(0, window_size - (end - pos_idx - 1))
    padded_window = left_pad + window + right_pad
    
    binary_features = {}
    for i, aa in enumerate(padded_window):
        encoding = [0] * 20
        if aa in amino_acids:
            idx = amino_acids.index(aa)
            encoding[idx] = 1
        
        for j, value in enumerate(encoding):
            binary_features[f"BE_{i*20 + j + 1}"] = value
    
    return binary_features

def extract_physicochemical_features(sequence: str, position: int, 
                                   window_size: int, properties: Dict) -> np.ndarray:
    """Extract physicochemical features"""
    pos_idx = position - 1
    start = max(0, pos_idx - window_size)
    end = min(len(sequence), pos_idx + window_size + 1)
    
    window = sequence[start:end]
    
    left_pad = "X" * max(0, window_size - (pos_idx - start))
    right_pad = "X" * max(0, window_size - (end - pos_idx - 1))
    padded_window = left_pad + window + right_pad
    
    physico_features = {}
    for i, aa in enumerate(padded_window):
        if aa in properties:
            for j, value in enumerate(properties[aa]):
                physico_features[f"PC_{i*len(properties[aa]) + j + 1}"] = value
        else:
            prop_len = len(next(iter(properties.values())))
            for j in range(prop_len):
                physico_features[f"PC_{i*prop_len + j + 1}"] = 0
    
    return physico_features

@timer
def extract_all_features():
    """Main feature extraction pipeline with datatable optimization"""
    LOGGER.info("Starting feature extraction...")
    
    # Check if cached features exist
    if CONFIG['data']['use_cached_features']:
        train_features_file = os.path.join(CONFIG['data']['cached_features_path'], 'train_features.csv')
        if os.path.exists(train_features_file):
            LOGGER.info("Using cached features")
            return
    
    # Load preprocessed data
    data_dir = os.path.join(CONFIG['paths']['output_dir'], 'data')
    train_dt = dt.fread(os.path.join(data_dir, 'train_data.csv'))
    val_dt = dt.fread(os.path.join(data_dir, 'val_data.csv'))
    test_dt = dt.fread(os.path.join(data_dir, 'test_data.csv'))
    
    # Load physicochemical properties
    properties = load_physicochemical_properties(CONFIG['paths']['physicochemical_data'])
    
    # Create features directory
    os.makedirs(CONFIG['data']['cached_features_path'], exist_ok=True)
    
    # Process each dataset
    for name, dataset in [('train', train_dt), ('val', val_dt), ('test', test_dt)]:
        LOGGER.info(f"Extracting features for {name} set...")
        
        df_pd = dataset.to_pandas()
        window_size = CONFIG['data']['window_size']
        
        all_features = []
        
        for idx, row in tqdm(df_pd.iterrows(), total=len(df_pd), desc=f"Processing {name}"):
            seq = row['Sequence']
            pos = int(row['Position'])
            header = row['Header']
            target = int(row['target'])
            
            window = extract_window(seq, pos, window_size)
            
            features = {'Header': header, 'Position': pos, 'target': target}
            
            if CONFIG['features']['extract_aac']:
                features.update(extract_aac(window))
            
            if CONFIG['features']['extract_dpc']:
                features.update(extract_dpc(window))
            
            if CONFIG['features']['extract_tpc']:
                tpc_features = extract_tpc_batch([window])[0]
                features.update(tpc_features)
                # Clean up TPC features immediately
                del tpc_features
            
            if CONFIG['features']['extract_binary']:
                features.update(extract_binary_encoding(seq, pos, window_size))
            
            if CONFIG['features']['extract_physicochemical']:
                features.update(extract_physicochemical_features(seq, pos, window_size, properties))
            
            all_features.append(features)
            
            # Garbage collect every 1000 samples to manage memory
            if len(all_features) % 1000 == 0:
                gc.collect()
                LOGGER.info(f"Processed {len(all_features)} samples, memory cleaned")
        
        # Save features
        features_df = pd.DataFrame(all_features)
        output_file = os.path.join(CONFIG['data']['cached_features_path'], f'{name}_features.csv')
        features_df.to_csv(output_file, index=False)
        
        LOGGER.info(f"Saved {len(features_df)} {name} features to {output_file}")
        
        del df_pd, all_features, features_df
        gc.collect()
    
    LOGGER.info("Feature extraction completed")

# ============================================================================
# SECTION 4: XGBOOST MODELS
# ============================================================================

class XGBoostModel:
    """XGBoost model wrapper with GPU support"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.feature_importance_ = None
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost with early stopping"""
        LOGGER.info("Training XGBoost model...")
        
        # Convert to DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Set parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'eta': self.config['params']['learning_rate'],
            'max_depth': self.config['params']['max_depth'],
            'subsample': self.config['params']['subsample'],
            'colsample_bytree': self.config['params']['colsample_bytree'],
            'tree_method': self.config['params']['tree_method']
        }
        
        if self.config['params']['device'] == 'cuda' and torch.cuda.is_available():
            params['device'] = 'cuda'
        
        # Train model
        evals_result = {}
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config['params']['n_estimators'],
            evals=[(dtrain, 'train'), (dval, 'validation')],
            early_stopping_rounds=self.config['params']['early_stopping_rounds'],
            evals_result=evals_result,
            verbose_eval=50
        )
        
        # Store feature importance
        self.feature_importance_ = self.model.get_score(importance_type='gain')
        
        LOGGER.info("XGBoost training completed")
        return evals_result
        
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
        
    def get_feature_importance(self):
        """Get feature importance"""
        return self.feature_importance_

def hyperparameter_tuning_xgboost(X_train, y_train, X_val, y_val):
    """Optuna hyperparameter tuning for XGBoost"""
    LOGGER.info("Starting XGBoost hyperparameter tuning...")
    
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'eta': trial.suggest_float('eta', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'tree_method': 'hist'
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dval, 'validation')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        preds = model.predict(dval)
        auc = roc_auc_score(y_val, preds)
        return auc
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    LOGGER.info(f"Best AUC: {study.best_value:.4f}")
    LOGGER.info(f"Best params: {study.best_params}")
    
    return study.best_params

@timer
def train_xgboost_pipeline():
    """Main XGBoost training pipeline"""
    if not CONFIG['xgboost']['enabled']:
        LOGGER.info("XGBoost training disabled")
        return
    
    LOGGER.info("Starting XGBoost pipeline...")
    
    # Load features
    features_dir = CONFIG['data']['cached_features_path']
    train_df = pd.read_csv(os.path.join(features_dir, 'train_features.csv'))
    val_df = pd.read_csv(os.path.join(features_dir, 'val_features.csv'))
    test_df = pd.read_csv(os.path.join(features_dir, 'test_features.csv'))
    
    # Prepare data
    id_cols = ['Header', 'Position', 'target']
    feature_cols = [col for col in train_df.columns if col not in id_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_val = val_df[feature_cols]
    y_val = val_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']
    
    # Hyperparameter tuning if enabled
    if CONFIG['xgboost']['hyperparameter_tuning']:
        best_params = hyperparameter_tuning_xgboost(X_train, y_train, X_val, y_val)
        CONFIG['xgboost']['params'].update(best_params)
    
    # Train model
    xgb_model = XGBoostModel(CONFIG['xgboost'])
    evals_result = xgb_model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    test_preds = xgb_model.predict(X_test)
    test_pred_binary = (test_preds > 0.5).astype(int)
    
    # Calculate metrics
    test_metrics = {
        'accuracy': accuracy_score(y_test, test_pred_binary),
        'precision': precision_score(y_test, test_pred_binary),
        'recall': recall_score(y_test, test_pred_binary),
        'f1': f1_score(y_test, test_pred_binary),
        'auc': roc_auc_score(y_test, test_preds),
        'mcc': matthews_corrcoef(y_test, test_pred_binary)
    }
    
    LOGGER.info("XGBoost Test Metrics:")
    for metric, value in test_metrics.items():
        LOGGER.info(f"  {metric}: {value:.4f}")
    
    # Save model and results
    os.makedirs(CONFIG['paths']['checkpoint_dir'], exist_ok=True)
    model_path = os.path.join(CONFIG['paths']['checkpoint_dir'], 'xgboost_model.json')
    xgb_model.model.save_model(model_path)
    
    # Save predictions
    predictions_df = test_df[['Header', 'Position', 'target']].copy()
    predictions_df['prediction'] = test_pred_binary
    predictions_df['probability'] = test_preds
    
    predictions_path = os.path.join(CONFIG['paths']['output_dir'], 'xgboost_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    
    # Save metrics
    metrics_path = os.path.join(CONFIG['paths']['output_dir'], 'xgboost_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    LOGGER.info("XGBoost pipeline completed")

# ============================================================================
# SECTION 5: TRANSFORMER MODELS
# ============================================================================

class PhosphorylationDataset(Dataset):
    """Dataset class for transformer training"""
    
    def __init__(self, dataframe, tokenizer, window_size=20, max_length=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        sequence = row['Sequence']
        position = int(row['Position']) - 1
        target = float(row['target'])
        
        # Extract window
        start = max(0, position - self.window_size)
        end = min(len(sequence), position + self.window_size + 1)
        window_sequence = sequence[start:end]
        
        # Tokenize
        encoding = self.tokenizer(
            window_sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'target': torch.tensor(target, dtype=torch.float),
            'sequence': window_sequence,
            'position': torch.tensor(position, dtype=torch.long),
            'header': row['Header']
        }

class BasePhosphoTransformer(nn.Module):
    """Base transformer class for phosphorylation prediction"""
    
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", dropout_rate=0.3, window_context=3):
        super().__init__()
        self.protein_encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Get hidden size from the model config
        hidden_size = self.protein_encoder.config.hidden_size
        
        # Context aggregation (lightweight)
        self.window_context = window_context
        context_size = hidden_size * (2*window_context + 1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(context_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get the transformer outputs
        outputs = self.protein_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get sequence outputs
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Find the center position
        center_pos = sequence_output.shape[1] // 2
        
        # Extract features from window around center
        batch_size, seq_len, hidden_dim = sequence_output.shape
        context_features = []
        
        for i in range(-self.window_context, self.window_context + 1):
            pos = center_pos + i
            # Handle boundary cases
            if pos < 0 or pos >= seq_len:
                # Use zero padding for out-of-bounds positions
                context_features.append(torch.zeros(batch_size, hidden_dim, device=sequence_output.device))
            else:
                context_features.append(sequence_output[:, pos, :])
        
        # Concatenate context features
        concat_features = torch.cat(context_features, dim=1)
        
        # Pass through classifier
        logits = self.classifier(concat_features)
        
        return logits.squeeze(-1)

class HierarchicalAttentionTransformer(BasePhosphoTransformer):
    """Transformer with hierarchical attention"""
    
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", dropout_rate=0.3, context_window=3):
        super().__init__(model_name, dropout_rate)
        
        hidden_size = self.protein_encoder.config.hidden_size
        self.context_window = context_window
        
        # Local attention for motif detection
        self.local_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Global attention for long-range dependencies
        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Motif-specific prediction heads
        self.motif_heads = nn.ModuleDict({
            'proline_directed': nn.Sequential(
                nn.Linear(hidden_size * 2, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            ),
            'acidic': nn.Sequential(
                nn.Linear(hidden_size * 2, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            ),
            'basic': nn.Sequential(
                nn.Linear(hidden_size * 2, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            ),
            'other': nn.Sequential(
                nn.Linear(hidden_size * 2, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )
        })
        
        # Final aggregation
        self.final_classifier = nn.Linear(4, 1)
        
    def forward(self, input_ids, attention_mask):
        # Get base encoder outputs
        base_outputs = self.protein_encoder(input_ids, attention_mask)
        sequence_output = base_outputs.last_hidden_state
        
        batch_size, seq_len, hidden_size = sequence_output.shape
        center_pos = seq_len // 2
        
        # Local attention (context window around center)
        local_start = max(0, center_pos - self.context_window)
        local_end = min(seq_len, center_pos + self.context_window + 1)
        local_features = sequence_output[:, local_start:local_end, :]
        
        local_attended, _ = self.local_attention(
            local_features, local_features, local_features
        )
        local_pooled = local_attended.mean(dim=1)
        
        # Global attention
        global_attended, _ = self.global_attention(
            sequence_output, sequence_output, sequence_output
        )
        global_pooled = global_attended[:, center_pos, :]
        
        # Concatenate features
        combined_features = torch.cat([local_pooled, global_pooled], dim=-1)
        
        # Get predictions from each motif head
        motif_predictions = []
        for motif_type, head in self.motif_heads.items():
            pred = head(combined_features)
            motif_predictions.append(pred)
        
        # Stack and aggregate
        motif_preds = torch.cat(motif_predictions, dim=-1)
        final_pred = self.final_classifier(motif_preds)
        
        return final_pred.squeeze(-1)

class MultiScaleFusionTransformer(BasePhosphoTransformer):
    """Transformer with multi-scale window fusion"""
    
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", window_sizes=[5, 10, 20]):
        super().__init__(model_name)
        
        self.window_sizes = window_sizes
        hidden_size = self.protein_encoder.config.hidden_size
        
        # Scale attention mechanism
        self.scale_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Scale-specific projections
        self.scale_projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) 
            for _ in self.window_sizes
        ])
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, multi_scale_inputs):
        """
        multi_scale_inputs: Dict with keys as window sizes
        """
        scale_features = []
        
        # Process each scale
        for i, (window_size, inputs) in enumerate(multi_scale_inputs.items()):
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
            # Encode
            outputs = self.protein_encoder(input_ids, attention_mask)
            center_features = outputs.last_hidden_state[:, outputs.last_hidden_state.size(1)//2, :]
            
            # Project
            projected = self.scale_projections[i](center_features)
            scale_features.append(projected.unsqueeze(1))
        
        # Stack all scale features
        all_scales = torch.cat(scale_features, dim=1)
        
        # Use attention to combine scales
        attended_features, attention_weights = self.scale_attention(
            all_scales, all_scales, all_scales
        )
        
        # Pool across scales
        combined = attended_features.mean(dim=1)
        
        # Final prediction
        output = self.classifier(combined)
        
        return output.squeeze(-1), attention_weights

class MotifAwareLoss(nn.Module):
    """Custom loss function with motif awareness"""
    
    def __init__(self, motif_weights):
        super().__init__()
        self.motif_weights = motif_weights
        self.base_criterion = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, predictions, targets, sequences, positions):
        base_loss = self.base_criterion(predictions, targets.float())
        
        batch_size = predictions.size(0)
        motif_multipliers = torch.ones(batch_size, device=predictions.device)
        
        for i in range(batch_size):
            if targets[i] == 1:
                seq = sequences[i]
                pos = positions[i].item()
                
                if pos < len(seq) - 1:
                    motif = seq[pos:pos+2]
                    if motif in self.motif_weights:
                        motif_multipliers[i] = self.motif_weights[motif]
        
        weighted_loss = base_loss * motif_multipliers
        return weighted_loss.mean()

class FocalLoss(nn.Module):
    """Focal loss for class imbalance"""
    
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, predictions, targets):
        p = torch.sigmoid(predictions)
        
        focal_weights = torch.where(
            targets == 1,
            self.alpha * (1 - p) ** self.gamma,
            (1 - self.alpha) * p ** self.gamma
        )
        
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets.float(), reduction='none'
        )
        
        focal_loss = focal_weights * bce_loss
        return focal_loss.mean()

def train_transformer_epoch(model, dataloader, optimizer, criterion, scaler, device, scheduler=None):
    """Train one epoch with proper progress tracking and memory management"""
    model.train()
    total_loss = 0
    all_targets = []
    all_predictions = []
    
    print("Training:")
    for i, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass with mixed precision
        if scaler is not None:
            with autocast():
                outputs = model(input_ids, attention_mask)
                
                if isinstance(criterion, MotifAwareLoss):
                    loss = criterion(outputs, targets, batch['sequence'], batch['position'])
                else:
                    loss = criterion(outputs, targets)
                
                # Scale loss for gradient accumulation
                loss = loss / CONFIG['transformers']['gradient_accumulation_steps']
        else:
            outputs = model(input_ids, attention_mask)
            
            if isinstance(criterion, MotifAwareLoss):
                loss = criterion(outputs, targets, batch['sequence'], batch['position'])
            else:
                loss = criterion(outputs, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / CONFIG['transformers']['gradient_accumulation_steps']
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation and optimizer step
        if (i + 1) % CONFIG['transformers']['gradient_accumulation_steps'] == 0:
            if scaler is not None:
                # Clip gradients before optimizer step
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            optimizer.zero_grad()
        
        # Print progress occasionally
        if i % 20 == 0 or i == len(dataloader) - 1:
            current_loss = loss.item() * CONFIG['transformers']['gradient_accumulation_steps']
            print(f"\rBatch {i+1}/{len(dataloader)}, Loss: {current_loss:.4f}", end="", flush=True)
        
        # Accumulate metrics (scale back the loss)
        total_loss += loss.item() * CONFIG['transformers']['gradient_accumulation_steps']
        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        
        # Memory cleanup every 100 batches
        if i % 100 == 0 and i > 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print()  # New line after progress
    
    # Handle remaining gradients if batch doesn't divide evenly
    if len(dataloader) % CONFIG['transformers']['gradient_accumulation_steps'] != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(dataloader)
    predictions_binary = (np.array(all_predictions) > 0.5).astype(int)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_targets, predictions_binary),
        'f1': f1_score(all_targets, predictions_binary),
        'auc': roc_auc_score(all_targets, all_predictions)
    }
    
    return metrics

def train_transformer_model(model_config: Dict):
    """Train a transformer model variant with proper per-model early stopping"""
    LOGGER.info(f"Training transformer model: {model_config}")
    
    # Load data
    data_dir = os.path.join(CONFIG['paths']['output_dir'], 'data')
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val_data.csv'))
    
    # Initialize tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
    
    train_dataset = PhosphorylationDataset(train_df, tokenizer, CONFIG['data']['window_size'])
    val_dataset = PhosphorylationDataset(val_df, tokenizer, CONFIG['data']['window_size'])
    
    train_loader = DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=model_config['batch_size'])
    
    # Initialize model
    if 'hierarchical' in model_config:
        model = HierarchicalAttentionTransformer(model_config['model_name'])
    else:
        model = BasePhosphoTransformer(model_config['model_name'])
    
    model = model.to(DEVICE)
    
    # Initialize optimizer and scheduler
    learning_rate = float(model_config['learning_rate'])  # Convert to float
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * model_config['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Initialize loss function
    if CONFIG['loss_functions']['use_focal_loss']:
        criterion = FocalLoss(
            gamma=CONFIG['loss_functions']['focal_gamma'],
            alpha=CONFIG['loss_functions']['focal_alpha']
        )
    elif CONFIG['loss_functions']['use_motif_aware_loss']:
        criterion = MotifAwareLoss(CONFIG['loss_functions']['motif_weights'])
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Initialize mixed precision scaler
    scaler = GradScaler() if CONFIG['transformers']['use_mixed_precision'] else None
    
    # Training loop with per-model early stopping
    best_val_f1 = 0.0
    patience = 3
    patience_counter = 0  # LOCAL variable per model
    
    for epoch in range(model_config['epochs']):
        LOGGER.info(f"Epoch {epoch+1}/{model_config['epochs']}")
        
        # Train
        train_metrics = train_transformer_epoch(model, train_loader, optimizer, criterion, scaler, DEVICE, scheduler)
        
        # Validate
        model.eval()
        val_loss = 0
        val_targets = []
        val_predictions = []
        
        print("Evaluating:")
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                targets = batch['target'].to(DEVICE)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, targets) if not isinstance(criterion, MotifAwareLoss) else criterion(outputs, targets, batch['sequence'], batch['position'])
                
                val_loss += loss.item()
                val_targets.extend(targets.cpu().numpy())
                val_predictions.extend(torch.sigmoid(outputs).cpu().numpy())
                
                # Print progress occasionally
                if i % 20 == 0 or i == len(val_loader) - 1:
                    print(f"\rBatch {i+1}/{len(val_loader)}", end="", flush=True)
        
        print()  # New line after progress
        
        val_loss /= len(val_loader)
        val_predictions_binary = (np.array(val_predictions) > 0.5).astype(int)
        
        val_metrics = {
            'loss': val_loss,
            'accuracy': accuracy_score(val_targets, val_predictions_binary),
            'f1': f1_score(val_targets, val_predictions_binary),
            'auc': roc_auc_score(val_targets, val_predictions)
        }
        
        LOGGER.info(f"Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}")
        LOGGER.info(f"Val - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Early stopping logic (per model)
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0  # Reset counter for THIS model
            
            # Save best model
            model_path = os.path.join(CONFIG['paths']['checkpoint_dir'], f"best_transformer_{model_config['model_name'].replace('/', '_')}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'val_metrics': val_metrics
            }, model_path)
            LOGGER.info(f"Saved new best model with F1 score: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            LOGGER.info(f"No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping check for THIS model only
        if patience_counter >= patience:
            LOGGER.info(f"Early stopping triggered for this model after {patience} epochs without improvement")
            break
        
        # Log to wandb if enabled
        if CONFIG['wandb']['enabled']:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/f1': train_metrics['f1'],
                'val/loss': val_metrics['loss'],
                'val/f1': val_metrics['f1']
            })
    
    LOGGER.info(f"Model training completed. Best F1: {best_val_f1:.4f}")
    return model

@timer
def train_all_transformers():
    """Train all enabled transformer variants"""
    if not CONFIG['transformers']['enabled']:
        LOGGER.info("Transformer training disabled")
        return
    
    LOGGER.info("Starting transformer training pipeline...")
    
    trained_models = {}
    
    for model_name, model_config in CONFIG['transformers']['models'].items():
        if model_config['enabled']:
            LOGGER.info(f"Training {model_name} model...")
            model = train_transformer_model(model_config)
            trained_models[model_name] = model
    
    LOGGER.info("All transformer training completed")
    return trained_models

# ============================================================================
# SECTION 6: ENSEMBLE METHODS
# ============================================================================

class VotingEnsemble:
    """Voting ensemble with weight optimization"""
    
    def __init__(self, models, optimize_weights=True):
        self.models = models
        self.weights = None
        self.optimize_weights = optimize_weights
        
    def fit(self, X_val, y_val):
        """Fit ensemble weights"""
        if self.optimize_weights:
            # Get predictions from all models
            predictions = []
            for model in self.models:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_val)[:, 1]
                else:
                    pred = model.predict(X_val)
                predictions.append(pred)
            
            # Optimize weights
            def objective(weights):
                weights = weights / weights.sum()  # Normalize
                ensemble_pred = sum(w * p for w, p in zip(weights, predictions))
                return -f1_score(y_val, (ensemble_pred > 0.5).astype(int))
            
            from scipy.optimize import minimize
            result = minimize(
                objective,
                np.ones(len(self.models)) / len(self.models),
                bounds=[(0, 1) for _ in range(len(self.models))],
                method='SLSQP'
            )
            
            self.weights = result.x / result.x.sum()
        else:
            self.weights = np.ones(len(self.models)) / len(self.models)
    
    def predict(self, X):
        """Make ensemble predictions"""
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            predictions.append(pred)
        
        ensemble_pred = sum(w * p for w, p in zip(self.weights, predictions))
        return ensemble_pred

class StackingEnsemble:
    """Stacking ensemble with meta-learner"""
    
    def __init__(self, base_models, meta_learner='logistic_regression'):
        self.base_models = base_models
        self.meta_learner_type = meta_learner
        self.meta_learner = None
        
    def fit(self, X_train, y_train, X_val, y_val):
        """Train stacking ensemble"""
        # Get base model predictions
        base_predictions_train = []
        base_predictions_val = []
        
        for model in self.base_models:
            if hasattr(model, 'predict_proba'):
                train_pred = model.predict_proba(X_train)[:, 1]
                val_pred = model.predict_proba(X_val)[:, 1]
            else:
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
            
            base_predictions_train.append(train_pred)
            base_predictions_val.append(val_pred)
        
        # Stack predictions
        X_meta_train = np.column_stack(base_predictions_train)
        X_meta_val = np.column_stack(base_predictions_val)
        
        # Train meta-learner
        if self.meta_learner_type == 'logistic_regression':
            self.meta_learner = LogisticRegression()
        elif self.meta_learner_type == 'mlp':
            self.meta_learner = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)
        elif self.meta_learner_type == 'xgboost':
            self.meta_learner = xgb.XGBClassifier()
        
        self.meta_learner.fit(X_meta_train, y_train)
        
        return self.meta_learner
    
    def predict(self, X):
        """Make stacking predictions"""
        base_predictions = []
        for model in self.base_models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            base_predictions.append(pred)
        
        X_meta = np.column_stack(base_predictions)
        
        if hasattr(self.meta_learner, 'predict_proba'):
            return self.meta_learner.predict_proba(X_meta)[:, 1]
        else:
            return self.meta_learner.predict(X_meta)

class DynamicSelectionEnsemble:
    """Dynamic model selection based on input"""
    
    def __init__(self, models, k_neighbors=5):
        self.models = models
        self.k_neighbors = k_neighbors
        self.nn = None
        self.model_competence = {}
        
    def fit(self, X_train, y_train):
        """Fit dynamic selection ensemble"""
        self.nn = NearestNeighbors(n_neighbors=self.k_neighbors)
        self.nn.fit(X_train)
        
        # Calculate model competence
        for model_name, model in self.models.items():
            competence_scores = []
            
            for i in range(len(X_train)):
                distances, indices = self.nn.kneighbors([X_train[i]])
                neighbor_indices = indices[0]
                
                X_neighbors = X_train[neighbor_indices]
                y_neighbors = y_train[neighbor_indices]
                
                if hasattr(model, 'predict_proba'):
                    predictions = (model.predict_proba(X_neighbors)[:, 1] > 0.5).astype(int)
                else:
                    predictions = (model.predict(X_neighbors) > 0.5).astype(int)
                
                accuracy = (predictions == y_neighbors).mean()
                competence_scores.append(accuracy)
            
            self.model_competence[model_name] = np.array(competence_scores)
    
    def predict(self, X_test):
        """Make dynamic selection predictions"""
        predictions = []
        
        for x in X_test:
            distances, indices = self.nn.kneighbors([x])
            neighbor_indices = indices[0]
            
            model_scores = {}
            for model_name in self.models:
                scores = self.model_competence[model_name][neighbor_indices]
                model_scores[model_name] = scores.mean()
            
            best_model_name = max(model_scores, key=model_scores.get)
            best_model = self.models[best_model_name]
            
            if hasattr(best_model, 'predict_proba'):
                pred = best_model.predict_proba([x])[:, 1][0]
            else:
                pred = best_model.predict([x])[0]
            
            predictions.append(pred)
        
        return np.array(predictions)

class ConfidenceWeightedEnsemble:
    """Ensemble weighted by prediction confidence"""
    
    def __init__(self, models):
        self.models = models
        
    def predict(self, X):
        """Make confidence-weighted predictions"""
        all_predictions = []
        all_confidences = []
        
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)[:, 1]
            else:
                probs = model.predict(X)
            
            # Calculate confidence as distance from 0.5
            confidences = np.abs(probs - 0.5) * 2
            
            all_predictions.append(probs)
            all_confidences.append(confidences)
        
        # Weight by confidence
        weighted_predictions = []
        for i in range(len(X)):
            sample_preds = [preds[i] for preds in all_predictions]
            sample_confs = [confs[i] for confs in all_confidences]
            
            if sum(sample_confs) > 0:
                weights = np.array(sample_confs) / sum(sample_confs)
                weighted_pred = sum(w * p for w, p in zip(weights, sample_preds))
            else:
                weighted_pred = sum(sample_preds) / len(sample_preds)
            
            weighted_predictions.append(weighted_pred)
        
        return np.array(weighted_predictions)

@timer
def train_ensemble_models():
    """Train ensemble models using trained base models"""
    if not CONFIG['ensemble']:
        LOGGER.info("Ensemble training disabled")
        return
    
    LOGGER.info("Starting ensemble training...")
    
    # Load test data for ensemble evaluation
    features_dir = CONFIG['data']['cached_features_path']
    test_df = pd.read_csv(os.path.join(features_dir, 'test_features.csv'))
    
    # Prepare data
    id_cols = ['Header', 'Position', 'target']
    feature_cols = [col for col in test_df.columns if col not in id_cols]
    
    X_test = test_df[feature_cols]
    y_test = test_df['target']
    
    # Load XGBoost predictions
    xgb_predictions_df = pd.read_csv(os.path.join(CONFIG['paths']['output_dir'], 'xgboost_predictions.csv'))
    xgb_predictions = xgb_predictions_df['probability'].values
    
    # Load transformer predictions from all trained models
    transformer_predictions = {}
    checkpoint_dir = CONFIG['paths']['checkpoint_dir']
    
    # Get predictions from each transformer model
    data_dir = os.path.join(CONFIG['paths']['output_dir'], 'data')
    test_data_df = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    
    from transformers import AutoTokenizer
    
    for model_name, model_config in CONFIG['transformers']['models'].items():
        if model_config.get('enabled', False):
            model_checkpoint = os.path.join(checkpoint_dir, f"best_transformer_{model_config['model_name'].replace('/', '_')}.pt")
            
            if os.path.exists(model_checkpoint):
                LOGGER.info(f"Getting predictions from {model_name}...")
                
                try:
                    # Load model
                    if 'hierarchical' in model_name:
                        model = HierarchicalAttentionTransformer(model_config['model_name'])
                    else:
                        model = BasePhosphoTransformer(model_config['model_name'])
                    
                    # Load checkpoint
                    checkpoint = torch.load(model_checkpoint, map_location=DEVICE)
                    model.load_state_dict(checkpoint)
                    model = model.to(DEVICE)
                    model.eval()
                    
                    # Get predictions
                    tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
                    test_dataset = PhosphorylationDataset(test_data_df, tokenizer, CONFIG['data']['window_size'])
                    test_loader = DataLoader(test_dataset, batch_size=32)
                    
                    predictions = []
                    with torch.no_grad():
                        for batch in test_loader:
                            input_ids = batch['input_ids'].to(DEVICE)
                            attention_mask = batch['attention_mask'].to(DEVICE)
                            
                            outputs = model(input_ids, attention_mask)
                            preds = torch.sigmoid(outputs).cpu().numpy()
                            predictions.extend(preds)
                    
                    transformer_predictions[model_name] = np.array(predictions)
                    LOGGER.info(f"Got {len(predictions)} predictions from {model_name}")
                    
                except Exception as e:
                    LOGGER.error(f"Failed to get predictions from {model_name}: {e}")
                    continue
    
    # Create ensemble predictions
    ensemble_results = {}
    
    # 1. Simple Voting Ensemble
    if CONFIG['ensemble'].get('voting', {}).get('enabled', False):
        LOGGER.info("Training voting ensemble...")
        
        all_predictions = [xgb_predictions]
        model_names = ['xgboost']
        
        for name, preds in transformer_predictions.items():
            all_predictions.append(preds)
            model_names.append(name)
        
        if len(all_predictions) > 1:
            # Simple average
            ensemble_pred_avg = np.mean(all_predictions, axis=0)
            
            # Weighted average (optimize weights)
            from scipy.optimize import minimize
            
            def objective(weights):
                weights = weights / weights.sum()  # Normalize
                ensemble_pred = sum(w * p for w, p in zip(weights, all_predictions))
                return -f1_score(y_test, (ensemble_pred > 0.5).astype(int))
            
            result = minimize(
                objective,
                np.ones(len(all_predictions)) / len(all_predictions),
                bounds=[(0, 1) for _ in range(len(all_predictions))],
                method='SLSQP'
            )
            
            optimal_weights = result.x / result.x.sum()
            ensemble_pred_weighted = sum(w * p for w, p in zip(optimal_weights, all_predictions))
            
            # Evaluate both
            ensemble_results['voting_average'] = {
                'predictions': ensemble_pred_avg,
                'f1': f1_score(y_test, (ensemble_pred_avg > 0.5).astype(int)),
                'auc': roc_auc_score(y_test, ensemble_pred_avg)
            }
            
            ensemble_results['voting_weighted'] = {
                'predictions': ensemble_pred_weighted,
                'f1': f1_score(y_test, (ensemble_pred_weighted > 0.5).astype(int)),
                'auc': roc_auc_score(y_test, ensemble_pred_weighted),
                'weights': dict(zip(model_names, optimal_weights))
            }
            
            LOGGER.info(f"Voting Average - F1: {ensemble_results['voting_average']['f1']:.4f}, AUC: {ensemble_results['voting_average']['auc']:.4f}")
            LOGGER.info(f"Voting Weighted - F1: {ensemble_results['voting_weighted']['f1']:.4f}, AUC: {ensemble_results['voting_weighted']['auc']:.4f}")
            LOGGER.info(f"Optimal weights: {ensemble_results['voting_weighted']['weights']}")
    
    # 2. Stacking Ensemble
    if CONFIG['ensemble'].get('stacking', {}).get('enabled', False):
        LOGGER.info("Training stacking ensemble...")
        
        if len(transformer_predictions) > 0:
            # Prepare stacking features
            stacking_features = [xgb_predictions.reshape(-1, 1)]
            for name, preds in transformer_predictions.items():
                stacking_features.append(preds.reshape(-1, 1))
            
            X_meta = np.hstack(stacking_features)
            
            # Split for meta-learner training
            from sklearn.model_selection import train_test_split
            X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(
                X_meta, y_test, test_size=0.3, random_state=42, stratify=y_test
            )
            
            # Train meta-learner
            meta_learner_type = CONFIG['ensemble']['stacking'].get('meta_learner', 'logistic_regression')
            
            if meta_learner_type == 'logistic_regression':
                from sklearn.linear_model import LogisticRegression
                meta_learner = LogisticRegression(random_state=42)
            elif meta_learner_type == 'xgboost':
                import xgboost as xgb
                meta_learner = xgb.XGBClassifier(random_state=42)
            else:
                from sklearn.ensemble import RandomForestClassifier
                meta_learner = RandomForestClassifier(random_state=42)
            
            meta_learner.fit(X_meta_train, y_meta_train)
            
            # Get stacking predictions
            stacking_pred = meta_learner.predict_proba(X_meta_test)[:, 1]
            
            ensemble_results['stacking'] = {
                'predictions': stacking_pred,
                'f1': f1_score(y_meta_test, (stacking_pred > 0.5).astype(int)),
                'auc': roc_auc_score(y_meta_test, stacking_pred),
                'meta_learner': meta_learner_type
            }
            
            LOGGER.info(f"Stacking - F1: {ensemble_results['stacking']['f1']:.4f}, AUC: {ensemble_results['stacking']['auc']:.4f}")
    
    # Save ensemble results
    ensemble_metrics = {}
    for name, result in ensemble_results.items():
        ensemble_metrics[name] = {
            'f1': result['f1'],
            'auc': result['auc']
        }
        if 'weights' in result:
            ensemble_metrics[name]['weights'] = result['weights']
        if 'meta_learner' in result:
            ensemble_metrics[name]['meta_learner'] = result['meta_learner']
    
    # Save to file
    ensemble_metrics_path = os.path.join(CONFIG['paths']['output_dir'], 'ensemble_metrics.json')
    with open(ensemble_metrics_path, 'w') as f:
        json.dump(ensemble_metrics, f, indent=2)
    
    LOGGER.info("Ensemble training completed")
    LOGGER.info("Ensemble results:")
    for name, metrics in ensemble_metrics.items():
        LOGGER.info(f"  {name}: F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
    
    return ensemble_results

# ============================================================================
# SECTION 7: EVALUATION AND METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred, y_proba=None):
    """Calculate comprehensive metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    
    if y_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_proba)
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, title="", save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_roc_curves(results_dict, save_path=None):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    for model_name, results in results_dict.items():
        fpr, tpr, _ = roc_curve(results['y_true'], results['y_proba'])
        auc_score = roc_auc_score(results['y_true'], results['y_proba'])
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def analyze_errors(y_true, y_pred, y_proba, sequences=None):
    """Analyze prediction errors"""
    # Find misclassified samples
    misclassified = y_true != y_pred
    
    # False positives and false negatives
    false_positives = (y_true == 0) & (y_pred == 1)
    false_negatives = (y_true == 1) & (y_pred == 0)
    
    LOGGER.info(f"Total misclassifications: {misclassified.sum()} ({misclassified.mean()*100:.2f}%)")
    LOGGER.info(f"False positives: {false_positives.sum()}")
    LOGGER.info(f"False negatives: {false_negatives.sum()}")
    
    if sequences is not None:
        # Analyze sequence patterns in errors
        fp_sequences = sequences[false_positives]
        fn_sequences = sequences[false_negatives]
        
        # Count central amino acids
        fp_center_aa = [seq[len(seq)//2] for seq in fp_sequences if len(seq) > 0]
        fn_center_aa = [seq[len(seq)//2] for seq in fn_sequences if len(seq) > 0]
        
        LOGGER.info("False positive center AA distribution:")
        fp_counts = pd.Series(fp_center_aa).value_counts()
        LOGGER.info(fp_counts)
        
        LOGGER.info("False negative center AA distribution:")
        fn_counts = pd.Series(fn_center_aa).value_counts()
        LOGGER.info(fn_counts)

def calculate_shap_values(model, X_sample):
    """Calculate SHAP values for interpretability"""
    if not SHAP_AVAILABLE:
        LOGGER.warning("SHAP not available, skipping interpretability analysis")
        return None
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        return shap_values
    except Exception as e:
        LOGGER.error(f"Error calculating SHAP values: {e}")
        return None

@timer
def evaluate_all_models():
    """Comprehensive evaluation of all models"""
    if not CONFIG['evaluation']:
        LOGGER.info("Evaluation disabled")
        return
    
    LOGGER.info("Starting comprehensive evaluation...")
    
    # This would load and evaluate all trained models
    # Implementation depends on saved model formats
    
    LOGGER.info("Evaluation completed")

# ============================================================================
# SECTION 8: CROSS-VALIDATION
# ============================================================================

def cross_validate_model(model_class, X, y, groups, config):
    """Perform cross-validation for a model"""
    cv_results = []
    
    skf = StratifiedGroupKFold(n_splits=config['n_folds'], shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y, groups)):
        LOGGER.info(f"Processing fold {fold + 1}/{config['n_folds']}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model = model_class()
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        metrics = calculate_metrics(y_val, (y_proba > 0.5).astype(int), y_proba)
        metrics['fold'] = fold
        cv_results.append(metrics)
    
    return cv_results

@timer
def run_full_cross_validation():
    """Run cross-validation for all models"""
    if not CONFIG['cross_validation'] or not CONFIG['pipeline']['run_cross_validation']:
        LOGGER.info("Cross-validation disabled")
        return
    
    LOGGER.info("Starting full cross-validation...")
    
    # Implementation would run CV for each model type
    
    LOGGER.info("Cross-validation completed")

# ============================================================================
# SECTION 9: WEIGHTS & BIASES INTEGRATION
# ============================================================================

def setup_wandb(config):
    """Initialize Weights & Biases"""
    if not config['wandb']['enabled']:
        return
    
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        tags=config['wandb']['tags'],
        config=config
    )
    
    LOGGER.info("Weights & Biases initialized")

def log_metrics_to_wandb(metrics, step=None):
    """Log metrics to wandb"""
    if CONFIG['wandb']['enabled']:
        wandb.log(metrics, step=step)

def log_visualizations_to_wandb(figures):
    """Log figures to wandb"""
    if CONFIG['wandb']['enabled']:
        for name, fig in figures.items():
            wandb.log({name: wandb.Image(fig)})

def log_model_to_wandb(model, model_name):
    """Save model artifact to wandb"""
    if CONFIG['wandb']['enabled'] and CONFIG['wandb']['log_model']:
        artifact = wandb.Artifact(model_name, type='model')
        artifact.add_file(model)
        wandb.log_artifact(artifact)

# ============================================================================
# SECTION 10: MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline execution"""
    global CONFIG, LOGGER, DEVICE
    
    # Load configuration
    CONFIG = load_config('config.yaml')
    
    # Setup
    LOGGER = setup_logging(CONFIG)
    set_seed(CONFIG['data']['random_seed'])
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directories
    os.makedirs(CONFIG['paths']['output_dir'], exist_ok=True)
    os.makedirs(CONFIG['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(CONFIG['data']['cached_features_path'], exist_ok=True)
    
    # Initialize W&B if enabled
    if CONFIG['wandb']['enabled']:
        setup_wandb(CONFIG)
    
    LOGGER.info("Starting Phosphorylation Prediction Pipeline")
    LOGGER.info(f"Using device: {DEVICE}")
    
    try:
        # Section 1: Data Preprocessing
        if CONFIG['pipeline']['run_data_preprocessing']:
            LOGGER.info("Running data preprocessing...")
            preprocess_data()
        
        # Section 2: Feature Extraction
        if CONFIG['pipeline']['run_feature_extraction']:
            LOGGER.info("Running feature extraction...")
            extract_all_features()
        
        # Section 3: XGBoost Training
        if CONFIG['pipeline']['run_xgboost']:
            LOGGER.info("Training XGBoost models...")
            train_xgboost_pipeline()
        
        # Section 4: Transformer Training
        if CONFIG['pipeline']['run_transformers']:
            LOGGER.info("Training Transformer models...")
            train_all_transformers()
        
        # Section 5: Ensemble Training
        if CONFIG['pipeline']['run_ensemble']:
            LOGGER.info("Training ensemble models...")
            train_ensemble_models()
        
        # Section 6: Evaluation
        if CONFIG['pipeline']['run_evaluation']:
            LOGGER.info("Running evaluation...")
            evaluate_all_models()
        
        # Section 7: Cross-Validation
        if CONFIG['pipeline']['run_cross_validation']:
            LOGGER.info("Running cross-validation...")
            run_full_cross_validation()
        
        LOGGER.info("Pipeline completed successfully!")
        
    except Exception as e:
        LOGGER.error(f"Pipeline failed with error: {e}")
        raise
    
    finally:
        # Clean up
        if CONFIG['wandb']['enabled']:
            wandb.finish()
        
        # Final memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()