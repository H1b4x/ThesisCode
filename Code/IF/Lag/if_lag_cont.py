import pandas as pd
import numpy as np
import json
import joblib
import logging
import gc
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pyarrow.parquet as pq

# Paths for loading and saving 
BASE_DIR = Path("/home/hajjohn1/bachelors-thesis")
TRAIN_PATH = BASE_DIR / "data" / "trainrf_lag.parquet"
EVAL_PATH  = BASE_DIR / "data" / "evalrf_lag.parquet"

RUN_DIR = BASE_DIR / "Clean" / "IF" / "lag" / f"run_continue_memopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_PATH     = RUN_DIR / "iforest_model.pkl"
SCALER_OUTPUT_PATH    = RUN_DIR / "scaler.pkl"
FEATURES_PATH         = RUN_DIR / "features.json"
RESULTS_PATH          = RUN_DIR / "results.json"
HYPERPARAMS_PATH      = RUN_DIR / "best_hyperparameters.json"
THRESHOLD_PATH        = RUN_DIR / "threshold.json"
CONFUSION_MATRIX_PATH = RUN_DIR / "confusion_matrix.npy"
LOG_FILE_PATH         = RUN_DIR / "training_continue_memopt.log"

# Fixed constants
SEED = 42
CONTAMINATION = 0.10
BETA = 0.5
GRID_PCTS = np.linspace(85.0, 95.0, 21)  # threshold percentile for grid search

# Previously selected/best hyperparameters (from earlier script run)
BEST_HP = {
    "n_estimators": 258,
    "max_samples": 0.6061695553391381,
    "max_features": 0.5909124836035503
}

# Logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def f_beta_corrected_stream(y_true, y_pred, beta=BETA):
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

# Load training data 
logger.info("Loading training data (only once)...")
train_df = pd.read_parquet(TRAIN_PATH)
if 'is_anomaly' not in train_df.columns:
    raise ValueError("Expected 'is_anomaly' column in training dataframe.")
features = [c for c in train_df.columns if c != 'is_anomaly']
train_df = train_df[features + ['is_anomaly']]

with open(FEATURES_PATH, "w") as f:
    json.dump(features, f)

# Scaler on trainig data
logger.info("Fitting StandardScaler on full training features and transforming (lag variant)...")
scaler = StandardScaler().fit(train_df[features].to_numpy(dtype=np.float32, copy=False))
X_train_full_scaled = scaler.transform(train_df[features].to_numpy(dtype=np.float32, copy=False))
y_train = train_df['is_anomaly'].to_numpy(dtype=np.int8)
joblib.dump(scaler, SCALER_OUTPUT_PATH)

# Prepare normal only training matrix from scaled data
mask_norm = (y_train == 0)
X_train_norm_scaled = X_train_full_scaled[mask_norm]

# Free memory from train_df and intermediates
del train_df, X_train_full_scaled, mask_norm
gc.collect()

# Train final model
logger.info("Training final IsolationForest model (memory-optimized) on scaled normals...")
final_model = IsolationForest(
    contamination=CONTAMINATION,
    n_estimators=int(BEST_HP["n_estimators"]),
    max_samples=float(BEST_HP["max_samples"]),
    max_features=float(BEST_HP["max_features"]),
    random_state=SEED,
    n_jobs=1 
)
final_model.fit(X_train_norm_scaled)

# Save model + HP
joblib.dump(final_model, MODEL_OUTPUT_PATH)
with open(HYPERPARAMS_PATH, 'w') as f:
    json.dump(BEST_HP, f, indent=2)
logger.info(f"Final model saved to {MODEL_OUTPUT_PATH}")

# Free training data
del X_train_norm_scaled, y_train
gc.collect()

# Evaluation
logger.info("Streaming evaluation data and computing anomaly scores (scaled)...")
scores_list = []
y_list = []

pqfile = pq.ParquetFile(str(EVAL_PATH))
batch_size = 100_000  # due to memory limits

for batch in pqfile.iter_batches(batch_size=batch_size, columns=features + ['is_anomaly']):
    batch_df = batch.to_pandas()
    y_batch = batch_df['is_anomaly'].to_numpy(dtype=np.int8)
    X_batch = batch_df[features].to_numpy(dtype=np.float32, copy=False)

    # transform with the saved scaler
    X_batch_scaled = scaler.transform(X_batch)
    scores_batch = -final_model.decision_function(X_batch_scaled)  

    scores_list.append(scores_batch)
    y_list.append(y_batch)

    del batch_df, X_batch, X_batch_scaled, y_batch
    gc.collect()

scores_eval = np.concatenate(scores_list)
y_eval = np.concatenate(y_list)
del scores_list, y_list
gc.collect()

# Threshold tuning using a lightweight grid search
logger.info("Tuning threshold via grid search over percentiles...")
best_f = -1.0
best_threshold_pct = None
best_threshold_value = None
for pct in GRID_PCTS:
    thresh = np.percentile(scores_eval, pct)
    y_pred = (scores_eval >= thresh).astype(int)
    f_beta, _, _ = f_beta_corrected_stream(y_eval, y_pred, beta=BETA)
    if f_beta > best_f:
        best_f = f_beta
        best_threshold_pct = float(pct)
        best_threshold_value = float(thresh)
logger.info(f"Selected percentile {best_threshold_pct} with threshold {best_threshold_value} (f_beta={best_f})")

# Final metrics/results
y_pred_eval = (scores_eval >= best_threshold_value).astype(int)
f_beta_final, precision_corr, recall_corr = f_beta_corrected_stream(y_eval, y_pred_eval, beta=BETA)
auc = roc_auc_score(y_eval, scores_eval) if len(np.unique(y_eval)) > 1 else 0.0
cm = confusion_matrix(y_eval, y_pred_eval)

results = {
    "f_beta_corrected": f_beta_final,
    "precision_corrected": precision_corr,
    "recall_corrected": recall_corr,
    "auc": auc,
    "confusion_matrix": cm.tolist(),
    "n_samples": int(len(y_eval)),
    "best_hyperparameters": BEST_HP,
    "threshold_percentile": best_threshold_pct,
    "threshold_value": best_threshold_value,
    "timestamp": datetime.now().isoformat()
}

# Save output
with open(RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=2)
np.save(CONFUSION_MATRIX_PATH, cm)
with open(THRESHOLD_PATH, 'w') as f:
    json.dump({"threshold_percentile": best_threshold_pct, "threshold_value": best_threshold_value}, f, indent=2)

logger.info(f"Completed memory-optimized continuation run (scaled). Artifacts in {RUN_DIR}")
