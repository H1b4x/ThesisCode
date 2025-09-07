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

def limit_memory(max_gb: float):
    """Limit memory usage of the script to `max_gb` GB."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    max_bytes = int(max_gb * 1024 ** 3)
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, hard))

try:
    limit_memory(85)
except Exception as e:
    print(f"WARNING: Could not set memory limit: {e}", file=sys.stderr)

# Set seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths
BASE_DIR = Path("/home/hajjohn1/bachelors-thesis")
TRAIN_PATH = BASE_DIR / "data" / "training.parquet"
EVAL_PATH = BASE_DIR / "data" / "evaluation.parquet"
TARGETS_PATH = BASE_DIR / "data" / "target_channels.csv"
RF_DIR = BASE_DIR / "final" / "RF" / "nolag_corrected"
RF_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RF_DIR / f"run_{timestamp}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_PATH = RUN_DIR / "rf_model.json"
LOG_FILE_PATH = RUN_DIR / "training.log"
RESULTS_PATH = RUN_DIR / "results.json"
HYPERPARAMS_PATH = RUN_DIR / "best_hyperparameters.json"
CONFUSION_MATRIX_PATH = RUN_DIR / "confusion_matrix.npy"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

N_TRIALS = 50
TIMEOUT_SECONDS = 60 * 60 * 24  # 24 hours

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
train_df = pd.read_parquet(TRAIN_PATH)
eval_df = pd.read_parquet(EVAL_PATH)
channels = pd.read_csv(TARGETS_PATH)["target_channels"].tolist()
logger.info(f"Training data: {train_df.shape}, Evaluation data: {eval_df.shape}")

def prune_correlated_features(df, feats, thresh=0.95):
    logger.info("Pruning highly correlated features (full training set)...")
    corr = df[feats].corr(numeric_only=True).abs()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    high = corr.where(mask).stack().loc[lambda x: x > thresh].index
    to_drop = set(f2 for f1, f2 in high)
    pruned_feats = [f for f in feats if f not in to_drop]
    logger.info(f"Pruned {len(feats) - len(pruned_feats)} features")
    return pruned_feats

input_feats = prune_correlated_features(train_df, channels)
logger.info(f"Continuing with {len(input_feats)} features...")

X_train = train_df[input_feats].values
y_train = train_df["is_anomaly"].values
X_eval = eval_df[input_feats].values
y_eval = eval_df["is_anomaly"].values

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

# TimeSeriesSplit for tuning
tscv = TimeSeriesSplit(n_splits=5)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': SEED,
        'n_jobs': -1
    }

    scores = []
    for train_index, val_index in tscv.split(X_train):
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]
        model = RandomForestClassifier(**params)
        model.fit(X_tr, y_tr)
        y_probs = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_model_comprehensive(y_val, y_probs)
        scores.append(metrics['f_beta_corrected'])
    return np.mean(scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS)

best = study.best_trial.params
logger.info(f"Best trial: {best}")

# Train the final model on the FULL training set
final_params = {
    **best,
    'random_state': SEED,
    'n_jobs': -1
}
final_model = RandomForestClassifier(**final_params)
final_model.fit(X_train, y_train)

# Evaluation
y_probs_final = final_model.predict_proba(X_eval)[:, 1]
final_metrics = evaluate_model_comprehensive(y_eval, y_probs_final)

# Save model as JSON-compatible dictionary
with open(MODEL_OUTPUT_PATH, 'w') as f:
    json.dump(final_model.get_params(), f, indent=2)

with open(HYPERPARAMS_PATH, 'w') as f:
    json.dump(best, f, indent=2)

with open(RESULTS_PATH, 'w') as f:
    json.dump(final_metrics, f, indent=2)

np.save(CONFUSION_MATRIX_PATH, np.array(final_metrics['confusion_matrix']))

logger.info(f"Training complete. Model and results saved in: {RUN_DIR}")
