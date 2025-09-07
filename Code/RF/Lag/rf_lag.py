import pandas as pd
import numpy as np
import optuna
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import logging
import json
from datetime import datetime
import random
import resource
import sys
import pyarrow.parquet as pq

def limit_memory(max_gb: float):
    """Limit memory usage of the script to `max_gb` GB."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    max_bytes = int(max_gb * 1024 ** 3)
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, hard))

try:
    limit_memory(85) #change the number 85 to allow more or less memory
except Exception as e:
    print(f"WARNING: Could not set memory limit: {e}", file=sys.stderr)

# Seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths
BASE_DIR = Path("/home/hajjohn1/bachelors-thesis")
TRAIN_PATH = BASE_DIR / "data" / "trainrf_lag.parquet"
EVAL_PATH = BASE_DIR / "data" / "evalrf_lag.parquet"
RF_DIR = BASE_DIR / "final" / "RandomForest" / "lag2"
RF_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RF_DIR / f"run_{timestamp}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_PATH = RUN_DIR / "rf_model.json"
LOG_FILE_PATH = RUN_DIR / "training.log"
RESULTS_PATH = RUN_DIR / "results.json"
HYPERPARAMS_PATH = RUN_DIR / "best_hyperparameters.json"
CONFUSION_MATRIX_PATH = RUN_DIR / "confusion_matrix.npy"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

N_TRIALS = 50
TIMEOUT_SECONDS = 60 * 60 * 24  

#Custom score function
def f_beta_corrected_stream(y_true, y_pred, beta=0.5):
    TP_e = FP_t = FN_e = N_t = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0:
            N_t += 1
        if yp == 1:
            if yt == 1:
                TP_e += 1
            else:
                FP_t += 1
        elif yt == 1:
            FN_e += 1
    precision_corr = (TP_e / (TP_e + FP_t) * (1 - FP_t / N_t)) if (TP_e + FP_t > 0 and N_t > 0) else 0.0
    recall = TP_e / (TP_e + FN_e) if (TP_e + FN_e) > 0 else 0.0
    b2 = beta * beta
    f_beta = ((1 + b2) * precision_corr * recall / (b2 * precision_corr + recall)) if (precision_corr + recall) > 0 else 0.0
    return f_beta, precision_corr, recall

logger.info("Loading training and evaluation data...")

# Function to read parquet files in chunks using pyarrow
def read_parquet_in_chunks(file_path, chunk_size):
    dataset = pq.ParquetDataset(file_path)
    table = dataset.read()
    df = table.to_pandas()
    return df

# Read train and eval data
train_df = read_parquet_in_chunks(TRAIN_PATH, chunk_size=10**6)
eval_df = read_parquet_in_chunks(EVAL_PATH, chunk_size=10**6)

logger.info(f"Training data: {train_df.shape}, Evaluation data: {eval_df.shape}")

features = [col for col in train_df.columns if col != "is_anomaly"]
logger.info(f"Using {len(features)} features...")

X_train = train_df[features].astype(np.float32).values
y_train = train_df["is_anomaly"].values
X_eval  = eval_df[features].astype(np.float32).values
y_eval  = eval_df["is_anomaly"].values

# Split training data
n_samples = X_train.shape[0]
half_index = n_samples // 2
# Tune on the first half of the data
X_tune, y_tune = X_train[:half_index], y_train[:half_index]
logger.info(f"Using first half of training data for hyperparameter tuning: {X_tune.shape[0]} samples.")

def evaluate_model_comprehensive(y_true, y_probs, beta=0.5):
    y_pred = (y_probs > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0
    cm = confusion_matrix(y_true, y_pred).tolist()
    f_beta, p_corr, r_corr = f_beta_corrected_stream(y_true, y_pred, beta)
    return {
        'f1': f1, 'precision': precision, 'recall': recall, 'auc': auc,
        'f_beta_corrected': f_beta,
        'precision_corrected': p_corr,
        'recall_corrected': r_corr,
        'confusion_matrix': cm,
        'n_samples': len(y_true)
    }

tscv = TimeSeriesSplit(n_splits=5)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 5),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': SEED,
        'n_jobs': -1
    }

    scores = []
    for train_idx, val_idx in tscv.split(X_tune):
        X_tr, X_val = X_tune[train_idx], X_tune[val_idx]
        y_tr, y_val = y_tune[train_idx], y_tune[val_idx]
        model = RandomForestClassifier(**params)
        model.fit(X_tr, y_tr)
        y_probs = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_model_comprehensive(y_val, y_probs)
        scores.append(metrics['f_beta_corrected'])
    return np.mean(scores)

# Run hyperparameter optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS)

best = study.best_trial.params
logger.info(f"Best trial: {best}")

# Final model: train on the entire training set
final_model = RandomForestClassifier(**best)
final_model.fit(X_train, y_train)

# Evaluation
y_probs_final = final_model.predict_proba(X_eval)[:, 1]
final_metrics = evaluate_model_comprehensive(y_eval, y_probs_final)

# Save outputs
with open(MODEL_OUTPUT_PATH, 'w') as f:
    json.dump(final_model.get_params(), f, indent=2)
with open(HYPERPARAMS_PATH, 'w') as f:
    json.dump(best, f, indent=2)
with open(RESULTS_PATH, 'w') as f:
    json.dump(final_metrics, f, indent=2)
np.save(CONFUSION_MATRIX_PATH, np.array(final_metrics['confusion_matrix']))

logger.info(f"Training complete. Model and results saved in: {RUN_DIR}")
