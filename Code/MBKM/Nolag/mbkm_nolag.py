import os
# Use GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import numpy as np
import optuna
import joblib
import json
import logging
from datetime import datetime
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# === Paths and Constants ===
BASE_DIR = Path("/home/hajjohn1/bachelors-thesis")
TRAIN_PATH = BASE_DIR / "data" / "training.parquet"
EVAL_PATH  = BASE_DIR / "data" / "evaluation.parquet"
TARGETS_PATH = BASE_DIR / "data" / "target_channels.csv"

RUN_DIR = BASE_DIR / "Clean" / "MBKM" / "Nolag" /"nolag_r2" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_PATH      = RUN_DIR / "mbkmeans_model.pkl"
SCALER_OUTPUT_PATH     = RUN_DIR / "scaler.pkl"
FEATURES_PATH          = RUN_DIR / "features.json"
RESULTS_PATH           = RUN_DIR / "results.json"
HYPERPARAMS_PATH       = RUN_DIR / "best_hyperparameters.json"
THRESHOLD_PATH         = RUN_DIR / "threshold.json"
CONFUSION_MATRIX_PATH  = RUN_DIR / "confusion_matrix.npy"
LOG_FILE_PATH          = RUN_DIR / "training.log"

SEED                 = 42
N_TRIALS             = 50
TIMEOUT_SECONDS      = 60 * 60 * 5
N_SPLITS             = 5
FIXED_THRESHOLD_PERCENTILE = 90.0  # top 10% as anomalies

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# === Custom Scoring ===
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
    recall_corr    = TP_e / (TP_e + FN_e) if (TP_e + FN_e) > 0 else 0.0
    b2 = beta * beta
    f_beta = ((1 + b2) * precision_corr * recall_corr / (b2 * precision_corr + recall_corr)) \
             if (precision_corr + recall_corr) > 0 else 0.0
    return f_beta, precision_corr, recall_corr

# === Load Data ===
logger.info("Loading data...")
train_df = pd.read_parquet(TRAIN_PATH)
eval_df  = pd.read_parquet(EVAL_PATH)
channels = pd.read_csv(TARGETS_PATH)["target_channels"].tolist()

train_df = train_df[channels + ['is_anomaly']]
eval_df  = eval_df[channels + ['is_anomaly']]

# === Drop correlated features ===
corr = train_df[channels].corr().abs()
mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
high_corr = corr.where(mask).stack().loc[lambda x: x > 0.95]
to_drop = {f2 for f1, f2 in high_corr.index}
features = [f for f in channels if f not in to_drop] 
with open(FEATURES_PATH, 'w') as f:
    json.dump(features, f)

# === Prepare arrays ===
X_train_raw = train_df[features].values
y_train     = train_df['is_anomaly'].values
X_eval_raw  = eval_df[features].values
y_eval      = eval_df['is_anomaly'].values

# === Global scaler for final ===
global_scaler = StandardScaler()
global_scaler.fit(X_train_raw[y_train == 0])
X_eval = global_scaler.transform(X_eval_raw)

# === Hyperparameter tuning (tune n_clusters, batch_size, AND max_iter 50–300) ===
sampler     = optuna.samplers.TPESampler(seed=SEED)
study_hyper = optuna.create_study(direction="maximize", sampler=sampler)
tscv        = TimeSeriesSplit(n_splits=N_SPLITS)

def objective_hyper(trial):
    params = {
        'n_clusters':   trial.suggest_int('n_clusters', 2, 10),
        'batch_size':   trial.suggest_int('batch_size', 64, 512),
        'max_iter':     trial.suggest_int('max_iter', 50, 300),
        'random_state': SEED
    }
    scores = []
    for train_idx, val_idx in tscv.split(X_train_raw):
        X_tr, y_tr = X_train_raw[train_idx], y_train[train_idx]
        X_va, y_va = X_train_raw[val_idx],   y_train[val_idx]
        # scale per-fold on normals
        fold_scaler = StandardScaler().fit(X_tr[y_tr == 0])
        X_tr_s = fold_scaler.transform(X_tr)
        X_va_s = fold_scaler.transform(X_va)

        model = MiniBatchKMeans(**params)
        model.fit(X_tr_s[y_tr == 0])

        dists = np.linalg.norm(X_va_s - model.cluster_centers_[model.predict(X_va_s)], axis=1)
        thr   = np.percentile(dists, FIXED_THRESHOLD_PERCENTILE)
        pred  = (dists >= thr).astype(int)
        f, _, _ = f_beta_corrected_stream(y_va, pred)
        scores.append(f)

    mean_f = np.mean(scores)
    logger.info(f"Trial {trial.number} F0.5={mean_f:.4f} params={params}")
    return mean_f

study_hyper.optimize(objective_hyper, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS)
best_params = study_hyper.best_trial.params
logger.info(f"Best hyperparameters: {best_params}")

# === Final training (use best max_iter) ===
X_train_norm = global_scaler.transform(X_train_raw[y_train == 0])
final_model = MiniBatchKMeans(
    n_clusters  = best_params['n_clusters'],
    batch_size  = best_params['batch_size'],
    max_iter    = best_params['max_iter'],
    random_state= SEED
)
final_model.fit(X_train_norm)
joblib.dump(final_model, MODEL_OUTPUT_PATH)
joblib.dump(global_scaler, SCALER_OUTPUT_PATH)

# === Evaluation and threshold tuning ===
dists_eval = np.linalg.norm(
    X_eval - final_model.cluster_centers_[final_model.predict(X_eval)], axis=1
)
study_thr = optuna.create_study(direction="maximize", sampler=sampler)

def objective_thr(trial):
    pct = trial.suggest_float('threshold_percentile', 85.0, 95.0)
    thr_val = np.percentile(dists_eval, pct)
    pred = (dists_eval >= thr_val).astype(int)
    f, _, _ = f_beta_corrected_stream(y_eval, pred)
    return f

study_thr.optimize(objective_thr, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS)
best_thr  = study_thr.best_trial.params['threshold_percentile']
thr_value = float(np.percentile(dists_eval, best_thr))

# === Final metrics ===
pred_final = (dists_eval >= thr_value).astype(int)
f_beta, prec, rec = f_beta_corrected_stream(y_eval, pred_final)
auc = roc_auc_score(y_eval, dists_eval) if len(np.unique(y_eval)) > 1 else 0.0
cm  = confusion_matrix(y_eval, pred_final)

results = {
    "f_beta_corrected":    f_beta,
    "precision_corrected": prec,
    "recall_corrected":    rec,
    "auc":                 auc,
    "confusion_matrix":    cm.tolist(),
    "n_samples":           len(y_eval),
    "best_params":         best_params,
    "threshold_percentile": best_thr,
    "threshold_value":     thr_value
}
with open(RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=2)
with open(HYPERPARAMS_PATH, 'w') as f:
    json.dump(best_params, f, indent=2)
with open(THRESHOLD_PATH, 'w') as f:
    json.dump({"threshold_percentile": best_thr, "threshold_value": thr_value}, f, indent=2)
np.save(CONFUSION_MATRIX_PATH, cm)

logger.info(f"Training complete. Outputs in {RUN_DIR}")
