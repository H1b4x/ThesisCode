import pandas as pd
import numpy as np
import random
import optuna
from optuna.samplers import TPESampler
import joblib
import json
import logging
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

#Paths and Constants
BASE_DIR = Path("/home/hajjohn1/bachelors-thesis")
TRAIN_PATH = BASE_DIR / "data" / "trainrf_lag.parquet"
EVAL_PATH = BASE_DIR / "data" / "evalrf_lag.parquet"

RUN_DIR = BASE_DIR / "final" / "IF" / "lag" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_PATH = RUN_DIR / "iforest_model.pkl"
FEATURES_PATH = RUN_DIR / "features.json"
RESULTS_PATH = RUN_DIR / "results.json"
HYPERPARAMS_PATH = RUN_DIR / "best_hyperparameters.json"
THRESHOLD_PATH = RUN_DIR / "threshold.json"
CONFUSION_MATRIX_PATH = RUN_DIR / "confusion_matrix.npy"
LOG_FILE_PATH = RUN_DIR / "training.log"

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_TRIALS_HP = 50        # Hyperparameter tuning trials
N_TRIALS_TH = 10        # Threshold tuning trials
TIMEOUT_SECONDS = 60 * 60 * 24 #24 hours time limit
N_SPLITS = 5
CONTAMINATION = 0.10    # consistent contamination for tuning
FIXED_PERCENTILE = (1 - CONTAMINATION) * 100

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

#Custom Score
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

# Load
logger.info("Loading data...")
train_df = pd.read_parquet(TRAIN_PATH)
eval_df = pd.read_parquet(EVAL_PATH)

# Use all columns except the label as features
if 'is_anomaly' not in train_df.columns or 'is_anomaly' not in eval_df.columns:
    raise ValueError("Expected 'is_anomaly' column in both training and evaluation dataframes.")

features = [c for c in train_df.columns if c != 'is_anomaly']

train_df = train_df[features + ['is_anomaly']]
eval_df = eval_df[features + ['is_anomaly']]

with open(FEATURES_PATH, "w") as f:
    json.dump(features, f)

# Hyperparameter Tuning with blockwise contiguous subsampling due to memory limits
# Parameters for blockwise subsample
TARGET_ROWS = 3_000_000
MAX_FRAC = 0.30  # ~30% of full train_df
hp_source_df = train_df.reset_index(drop=True)  # full data for sampling

# compute block size: min(target, 30% of available)
block_size = min(TARGET_ROWS, int(len(hp_source_df) * MAX_FRAC))
if block_size < 1000:
    raise ValueError("Block size too small for meaningful tuning.")

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

def sample_block(df, size, seed):
    rng = np.random.default_rng(seed)
    max_start = len(df) - size
    start = int(rng.integers(0, max_start + 1))
    return df.iloc[start : start + size].reset_index(drop=True)

# SAMPLE ONCE FOR ALL TRIALS
fixed_hp_df = sample_block(hp_source_df, block_size, seed=SEED)

def objective_hp(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
        'max_features': trial.suggest_float('max_features', 0.5, 1.0),
        'random_state': SEED,
        'n_jobs': -1
    }
    f_scores = []
    for train_idx, val_idx in tscv.split(fixed_hp_df):
        X_tr = fixed_hp_df.iloc[train_idx][features]
        y_tr = fixed_hp_df.iloc[train_idx]['is_anomaly']
        X_va = fixed_hp_df.iloc[val_idx][features]
        y_va = fixed_hp_df.iloc[val_idx]['is_anomaly']
        # train only on normal data
        X_train_norm = X_tr[y_tr == 0]
        model = IsolationForest(contamination=CONTAMINATION, **params)
        model.fit(X_train_norm)
        scores = -model.decision_function(X_va)
        thresh = np.percentile(scores, FIXED_PERCENTILE)
        y_pred = (scores >= thresh).astype(int)
        f_beta, _, _ = f_beta_corrected_stream(y_va, y_pred, beta=0.5)
        f_scores.append(f_beta)
    return np.mean(f_scores)

study_hp = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=SEED)
)
study_hp.optimize(objective_hp, n_trials=N_TRIALS_HP, timeout=TIMEOUT_SECONDS)
best_hp = study_hp.best_trial.params
logger.info(f"Best HP: {best_hp}")
with open(HYPERPARAMS_PATH, 'w') as f:
    json.dump(best_hp, f, indent=2)

#Train Final Model
X_train_full = train_df[features]
y_train_full = train_df['is_anomaly'].values

X_train_final = X_train_full[y_train_full == 0]
final_model = IsolationForest(
    contamination=CONTAMINATION,
    n_estimators=best_hp['n_estimators'],
    max_samples=best_hp['max_samples'],
    max_features=best_hp['max_features'],
    random_state=SEED,
    n_jobs=-1
)
final_model.fit(X_train_final)
joblib.dump(final_model, MODEL_OUTPUT_PATH)

# Threshold Tuning on Eval Set
X_eval = eval_df[features]
y_eval = eval_df['is_anomaly'].values

def objective_th(trial):
    pct = trial.suggest_float('threshold_percentile', 85.0, 95.0)
    scores = -final_model.decision_function(X_eval)
    thresh = np.percentile(scores, pct)
    y_pred = (scores >= thresh).astype(int)
    f_beta, _, _ = f_beta_corrected_stream(y_eval, y_pred, beta=0.5)
    return f_beta

study_th = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=SEED)
)
study_th.optimize(objective_th, n_trials=N_TRIALS_TH, timeout=TIMEOUT_SECONDS)
best_threshold_pct = study_th.best_trial.params['threshold_percentile']
scores_eval = -final_model.decision_function(X_eval)
threshold_value = np.percentile(scores_eval, best_threshold_pct)

# Final Evaluation
y_pred_eval = (scores_eval >= threshold_value).astype(int)
f_beta, precision_corr, recall_corr = f_beta_corrected_stream(y_eval, y_pred_eval, beta=0.5)
auc = roc_auc_score(y_eval, scores_eval) if len(np.unique(y_eval)) > 1 else 0.0
cm = confusion_matrix(y_eval, y_pred_eval)

results = {
    "f_beta_corrected": f_beta,
    "precision_corrected": precision_corr,
    "recall_corrected": recall_corr,
    "auc": auc,
    "confusion_matrix": cm.tolist(),
    "n_samples": len(y_eval),
    "best_hyperparameters": best_hp,
    "threshold_percentile": best_threshold_pct,
    "threshold_value": threshold_value
}
with open(RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=2)
np.save(CONFUSION_MATRIX_PATH, cm)
with open(THRESHOLD_PATH, 'w') as f:
    json.dump({"threshold_percentile": best_threshold_pct, "threshold_value": threshold_value}, f, indent=2)

logger.info(f"Completed two-stage tuning with contamination={CONTAMINATION}. Artifacts saved in {RUN_DIR}")
