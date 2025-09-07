import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pyarrow.dataset as ds
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
import gc

# === Paths and Constants ===
BASE_DIR      = Path("/home/hajjohn1/bachelors-thesis")
TRAIN_PATH    = BASE_DIR / "data" / "train_lag.parquet"
EVAL_PATH     = BASE_DIR / "data" / "eval_lag.parquet"

RUN_DIR       = BASE_DIR / "final" / "MBKM" / "lag2_r" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_PATH      = RUN_DIR / "mbkmeans_model.pkl"
SCALER_OUTPUT_PATH     = RUN_DIR / "scaler.pkl"
FEATURES_PATH          = RUN_DIR / "features.json"
RESULTS_PATH           = RUN_DIR / "results.json"
HYPERPARAMS_PATH       = RUN_DIR / "best_hyperparameters.json"
THRESHOLD_PATH         = RUN_DIR / "threshold.json"
CONFUSION_MATRIX_PATH  = RUN_DIR / "confusion_matrix.npy"
LOG_FILE_PATH          = RUN_DIR / "training.log"

SEED                    = 42
N_TRIALS                = 50
TIMEOUT_SECONDS         = 60 * 60 * 5
N_SPLITS                = 5
CHUNK_SIZE              = 100_000   # based on available memory
FIXED_THRESHOLD_PERCENTILE = 90.0  

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
    recall_corr    = TP_e / (TP_e + FN_e) if (TP_e + FN_e) > 0 else 0.0
    b2 = beta * beta
    f_beta = ((1 + b2) * precision_corr * recall_corr / (b2 * precision_corr + recall_corr)) \
             if (precision_corr + recall_corr) > 0 else 0.0
    return f_beta, precision_corr, recall_corr

# iterate Parquet in chunks
def parquet_row_batches(path, columns, batch_size):
    ds_obj = ds.dataset(str(path), format="parquet")
    for batch in ds_obj.to_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()

# Discover features once via PyArrow schema 
ds_obj = ds.dataset(str(TRAIN_PATH), format="parquet")
features = [col for col in ds_obj.schema.names if col != 'is_anomaly']
with open(FEATURES_PATH, 'w') as f:
    json.dump(features, f)

# Stream‐fit global scaler on normal data only
logger.info("Building global scaler via chunked partial_fit...")
global_scaler = StandardScaler()
for chunk in parquet_row_batches(TRAIN_PATH, features + ['is_anomaly'], CHUNK_SIZE):
    normals = chunk.loc[chunk['is_anomaly'] == 0, features].values.astype(np.float32)
    if normals.size:
        global_scaler.partial_fit(normals)
    del chunk, normals
    gc.collect()
joblib.dump(global_scaler, SCALER_OUTPUT_PATH)

# Load full datasets for hyperparameter tuning
logger.info("Loading full datasets for Optuna hyperparameter tuning...")
train_df = pd.read_parquet(TRAIN_PATH, columns=features + ['is_anomaly'])
eval_df  = pd.read_parquet(EVAL_PATH,  columns=features + ['is_anomaly'])

X_train_raw = train_df[features].values.astype(np.float32)
y_train     = train_df['is_anomaly'].values
X_eval_raw  = eval_df[features].values.astype(np.float32)
y_eval      = eval_df['is_anomaly'].values

del train_df, eval_df
gc.collect()

logger.info(f"Using {len(y_train)} samples for Optuna trials")

# Hyperparameter tuning
sampler     = optuna.samplers.TPESampler(seed=SEED)
study_hyper = optuna.create_study(direction="maximize", sampler=sampler)
tscv        = TimeSeriesSplit(n_splits=N_SPLITS)

def objective_hyper(trial):
    params = {
        'n_clusters':   trial.suggest_int('n_clusters',    2, 10),
        'batch_size':   trial.suggest_int('batch_size',   64, 512),
        'max_iter':     trial.suggest_int('max_iter',     50, 300),  # now tunable
        'random_state': SEED
    }
    scores = []
    for train_idx, val_idx in tscv.split(X_train_raw):
        X_tr, y_tr = X_train_raw[train_idx], y_train[train_idx]
        X_va, y_va = X_train_raw[val_idx],   y_train[val_idx]

        # per-fold scaling
        fold_scaler = StandardScaler().fit(X_tr[y_tr == 0])
        X_tr_s = fold_scaler.transform(X_tr).astype(np.float32)
        X_va_s = fold_scaler.transform(X_va).astype(np.float32)

        model = MiniBatchKMeans(**params)
        model.fit(X_tr_s[y_tr == 0])

        dists = np.linalg.norm(
            X_va_s - model.cluster_centers_[model.predict(X_va_s)],
            axis=1
        )
        thr   = np.percentile(dists, FIXED_THRESHOLD_PERCENTILE)
        pred  = (dists >= thr).astype(int)
        f, _, _ = f_beta_corrected_stream(y_va, pred)
        scores.append(f)

        del fold_scaler, X_tr_s, X_va_s, model, dists, pred
        gc.collect()

    mean_f = np.mean(scores)
    logger.info(f"Trial {trial.number} — mean F0.5: {mean_f:.4f}, params: {params}")
    return mean_f

study_hyper.optimize(objective_hyper, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS)
best_params = study_hyper.best_trial.params
logger.info(f"Best hyperparameters: {best_params}")
with open(HYPERPARAMS_PATH, 'w') as f:
    json.dump(best_params, f, indent=2)

# Final training with chunked MiniBatchKMeans.partial_fit 
logger.info("Final training via chunked MiniBatchKMeans...")
final_model = MiniBatchKMeans(
    n_clusters   = best_params['n_clusters'],
    batch_size   = best_params['batch_size'],
    max_iter     = best_params['max_iter'],
    random_state = SEED
)
for chunk in parquet_row_batches(TRAIN_PATH, features + ['is_anomaly'], CHUNK_SIZE):
    X_chunk = chunk.loc[chunk['is_anomaly'] == 0, features].values.astype(np.float32)
    X_scaled = global_scaler.transform(X_chunk)
    final_model.partial_fit(X_scaled)
    del chunk, X_chunk, X_scaled
    gc.collect()
joblib.dump(final_model, MODEL_OUTPUT_PATH)

# Chunked evaluation: compute distances & collect labels
logger.info("Evaluating via chunked distance computation...")
dists_list = []
labels_list = []
for chunk in parquet_row_batches(EVAL_PATH, features + ['is_anomaly'], CHUNK_SIZE):
    Xc = chunk[features].values.astype(np.float32)
    Xs = global_scaler.transform(Xc)
    d  = np.linalg.norm(
        Xs - final_model.cluster_centers_[final_model.predict(Xs)],
        axis=1
    )
    dists_list.append(d)
    labels_list.append(chunk['is_anomaly'].values)
    del chunk, Xc, Xs, d
    gc.collect()

dists_eval = np.concatenate(dists_list)
y_eval     = np.concatenate(labels_list)

# Threshold tuning 
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
with open(THRESHOLD_PATH, 'w') as f:
    json.dump({"threshold_percentile": best_thr, "threshold_value": thr_value}, f, indent=2)

# Final metrics
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
np.save(CONFUSION_MATRIX_PATH, cm)

logger.info(f"Training complete. All outputs in {RUN_DIR}")
