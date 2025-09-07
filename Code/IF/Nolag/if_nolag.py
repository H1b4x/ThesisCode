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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# === Paths and Constants ===
BASE_DIR = Path("/home/hajjohn1/bachelors-thesis")
TRAIN_PATH = BASE_DIR / "data" / "training.parquet"
EVAL_PATH = BASE_DIR / "data" / "evaluation.parquet"
TARGETS_PATH = BASE_DIR / "data" / "target_channels.csv"

RUN_DIR = BASE_DIR / "final" / "IF" / "nolag" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_PATH = RUN_DIR / "iforest_model.pkl"
SCALER_OUTPUT_PATH = RUN_DIR / "scaler.pkl"
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
TIMEOUT_SECONDS = 60 * 60 * 24
N_SPLITS = 5
CONTAMINATION = 0.10    # consistent contamination for both tuning and final model
FIXED_PERCENTILE = (1 - CONTAMINATION) * 100

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Custom score function
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

# Load & Preprocess
logger.info("Loading data...")
train_df = pd.read_parquet(TRAIN_PATH)
eval_df = pd.read_parquet(EVAL_PATH)
target_channels = pd.read_csv(TARGETS_PATH)["target_channels"].tolist()

train_df = train_df[target_channels + ['is_anomaly']]
eval_df = eval_df[target_channels + ['is_anomaly']]

# Drop highly correlated features
corr = train_df[target_channels].corr().abs()
mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
high_corr = corr.where(mask).stack().loc[lambda x: x > 0.995]
to_drop = set(f2 for f1, f2 in high_corr.index)
features = [f for f in target_channels if f not in to_drop]
with open(FEATURES_PATH, "w") as f:
    json.dump(features, f)

# Hyperparameter Tuning
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

def objective_hp(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
        'max_features': trial.suggest_float('max_features', 0.5, 1.0),
        'random_state': SEED,
        'n_jobs': -1
    }
    f_scores = []
    for train_idx, val_idx in tscv.split(train_df):
        # split raw data
        X_tr = train_df.iloc[train_idx][features]
        y_tr = train_df.iloc[train_idx]['is_anomaly']
        X_va = train_df.iloc[val_idx][features]
        y_va = train_df.iloc[val_idx]['is_anomaly']
        # fit scaler only on TRAIN fold to avoid leakage
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)
        # train only on normal data
        X_train_norm = X_tr_s[y_tr == 0]
        model = IsolationForest(contamination=CONTAMINATION, **params)
        model.fit(X_train_norm)
        scores = -model.decision_function(X_va_s)
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

# Train Final Model with Best Hyperparams on train with scaling
scaler = StandardScaler().fit(train_df[features])
X_train_full = scaler.transform(train_df[features])
y_train_full = train_df['is_anomaly'].values
joblib.dump(scaler, SCALER_OUTPUT_PATH)

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

# Threshold Tuning on Eval Set with same scaler
X_eval = scaler.transform(eval_df[features])
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

# Evaluation
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
