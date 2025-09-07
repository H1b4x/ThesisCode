import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import logging
import json
from datetime import datetime
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path("/home/hajjohn1/bachelors-thesis")
TRAIN_PATH = BASE_DIR / "data" / "train_lag.parquet"
EVAL_PATH = BASE_DIR / "data" / "eval_lag.parquet"
XGB_DIR = BASE_DIR / "Clean" / "XGBoost" / "Lag"
XGB_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = XGB_DIR / f"run_{timestamp}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_PATH = RUN_DIR / "xgb_model.json"
LOG_FILE_PATH = RUN_DIR / "training.log"
RESULTS_PATH = RUN_DIR / "results.json"
HYPERPARAMS_PATH = RUN_DIR / "best_hyperparameters.json"
CONFUSION_MATRIX_PATH = RUN_DIR / "confusion_matrix.npy"


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

N_TRIALS = 50
TIMEOUT_SECONDS = 60 * 60 * 5  # 5 hours


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
logger.info(f"Training data: {train_df.shape}, Evaluation data: {eval_df.shape}")

X_train = train_df.drop(columns=["is_anomaly"]).values
y_train = train_df["is_anomaly"].values
X_eval = eval_df.drop(columns=["is_anomaly"]).values
y_eval = eval_df["is_anomaly"].values

logger.info(f"Using {X_train.shape[1]} input features...")

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
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist',
        'gpu_id': 1,
        'random_state': SEED,
        'eta': trial.suggest_float('eta', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True)
    }

    scores = []

    for train_index, val_index in tscv.split(X_train):
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, "eval")],
            early_stopping_rounds=20,
            verbose_eval=False
        )

        y_probs = model.predict(dval)
        metrics = evaluate_model_comprehensive(y_val, y_probs)
        scores.append(metrics['f_beta_corrected'])

    return np.mean(scores)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS)


best = study.best_trial.params
logger.info(f"Best trial: {best}")


train_idx, val_idx = list(tscv.split(X_train))[-1]
X_tr, X_val = X_train[train_idx], X_train[val_idx]
y_tr, y_val = y_train[train_idx], y_train[val_idx]

dtrain_final = xgb.DMatrix(X_tr, label=y_tr)
dval_final = xgb.DMatrix(X_val, label=y_val)

final_params = {
    **best,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'gpu_hist',
    'gpu_id': 1,
    'random_state': SEED
}

final_model = xgb.train(
    final_params,
    dtrain_final,
    num_boost_round=500,
    evals=[(dval_final, "eval")],
    early_stopping_rounds=30,
    verbose_eval=True
)


deval = xgb.DMatrix(X_eval)
y_probs_final = final_model.predict(deval)
final_metrics = evaluate_model_comprehensive(y_eval, y_probs_final)


final_model.save_model(MODEL_OUTPUT_PATH)

with open(HYPERPARAMS_PATH, 'w') as f:
    json.dump(best, f, indent=2)

with open(RESULTS_PATH, 'w') as f:
    json.dump(final_metrics, f, indent=2)

np.save(CONFUSION_MATRIX_PATH, np.array(final_metrics['confusion_matrix']))

logger.info(f"Training complete. Model and results saved in: {RUN_DIR}")
