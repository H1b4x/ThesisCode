#!/usr/bin/env python3
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

# Optional memory guard 
try:
    import resource
    def limit_memory(max_gb: float):
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        max_bytes = int(max_gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, hard))
    try:
        limit_memory(85)
    except Exception as e:
        print(f"WARNING: Could not set memory limit: {e}", file=sys.stderr)
except Exception:
    pass

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR   = Path("/home/hajjohn1/bachelors-thesis")
TRAIN_PATH = BASE_DIR / "data" / "training.parquet"
EVAL_PATH  = BASE_DIR / "data" / "evaluation.parquet"
TARGETS_PATH = BASE_DIR / "data" / "target_channels.csv"

# Output under the same RF root as before
OUT_ROOT = BASE_DIR / "final" / "RF" / "nolag_corrected"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUT_ROOT / f"redo_plots_{timestamp}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_PATH       = RUN_DIR / "rf_model_params.json"  
FEATURES_PATH           = RUN_DIR / "input_features.json"
RESULTS_EVAL_PATH       = RUN_DIR / "results_eval.json"
RESULTS_TRAIN_PATH      = RUN_DIR / "results_train.json"
CONFUSION_MATRIX_EVAL   = RUN_DIR / "confusion_matrix_eval.npy"
CONFUSION_MATRIX_TRAIN  = RUN_DIR / "confusion_matrix_train.npy"
EVENTS_EVAL_JSON        = RUN_DIR / "events_eval.json"
EVENTS_TRAIN_JSON       = RUN_DIR / "events_train.json"
PLOT_COUNTS_EVAL        = RUN_DIR / "detected_vs_missed_eval.png"
PLOT_COUNTS_TRAIN       = RUN_DIR / "detected_vs_missed_train.png"
LOG_FILE_PATH           = RUN_DIR / "redo_plots.log"

# Known best hyperparameters (no Optuna rerun)
BEST_PARAMS = {
    "n_estimators": 375,
    "max_depth": 9,
    "min_samples_split": 5,
    "min_samples_leaf": 4,
    "max_features": "log2",
    "random_state": SEED,
    "n_jobs": -1
}

THRESHOLD = 0.5  # keep consistent with original RF evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()]
)
logger = logging.getLogger("rf_redo_plots")


def prune_correlated_features(df: pd.DataFrame, feats, thresh: float = 0.95):
    logger.info("Pruning highly correlated features (full training set)...")
    corr = df[feats].corr(numeric_only=True).abs()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    high = corr.where(mask).stack().loc[lambda x: x > thresh].index
    to_drop = set(f2 for f1, f2 in high)
    pruned_feats = [f for f in feats if f not in to_drop]
    logger.info(f"Pruned {len(feats) - len(pruned_feats)} features (kept {len(pruned_feats)})")
    return pruned_feats

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

def evaluate_model_comprehensive(y_true, y_probs, beta=0.5, threshold=0.5):
    y_pred = (y_probs > threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0
    cm = confusion_matrix(y_true, y_pred).tolist()
    f_beta, p_corr, r_corr = f_beta_corrected_stream(y_true, y_pred, beta)
    return {
        "threshold": threshold,
        "f1": f1, "precision": precision, "recall": recall, "auc": auc,
        "f_beta_corrected": f_beta,
        "precision_corrected": p_corr,
        "recall_corrected": r_corr,
        "confusion_matrix": cm,
        "n_samples": int(len(y_true))
    }, y_pred

def build_events(df: pd.DataFrame, binary_series: np.ndarray):
    d = df.copy()
    d["pred_anomaly"] = binary_series.astype(int)
    d["event_true"] = (d["is_anomaly"] == 1)

    mask = d["event_true"]
    event_starts = mask & (~mask.shift(fill_value=False))
    d["event_id"] = np.where(mask, event_starts.cumsum(), np.nan)

    events = []
    if mask.any():
        for eid, grp in d[mask].groupby("event_id"):
            length = int(len(grp))
            detected = bool(grp["pred_anomaly"].any())
            events.append({"event_id": int(eid), "length": length, "detected": detected})
    return events

def _nice_bins(max_len: int):
    candidates = [1, 1000, 10000, 20000, 30000, 60000, max_len + 1]
    bins = []
    for b in candidates:
        if not bins or b > bins[-1]:
            bins.append(b)
    if len(bins) < 2:
        bins = [1, max_len + 1]
    # build labels
    labels = []
    for i in range(len(bins) - 1):
        a, b = bins[i], bins[i + 1]
        def _fmt(x):
            return f"{x//1000}k" if x >= 1000 else f"{x}"
        labels.append(f"{_fmt(a)}–{_fmt(b-1)}")
    return bins, labels

def plot_detected_vs_missed(events, title, out_path):
    if not events:
        logger.info(f"{title}: no true events; skipping plot.")
        return

    df_ev = pd.DataFrame(events)[["length", "detected"]]
    max_len = int(df_ev["length"].max())
    bins, labels = _nice_bins(max_len)
    df_ev["length_bin"] = pd.cut(df_ev["length"], bins=bins, labels=labels, right=False, include_lowest=True)

    grouped = df_ev.groupby("length_bin", observed=True)["detected"].agg(total_events="count", detected_events="sum")
    undetected = grouped["total_events"] - grouped["detected_events"]

    x = np.arange(len(grouped.index))
    width = 0.4
    plt.figure(figsize=(9, 4.5))
    plt.bar(x - width/2, grouped["detected_events"], width, label="Detected")
    plt.bar(x + width/2, undetected, width, label="Missed")
    plt.xticks(x, grouped.index.astype(str), rotation=45, ha="right")
    plt.xlabel("Event Length Range")
    plt.ylabel("Number of Events")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    logger.info("Loading data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    eval_df  = pd.read_parquet(EVAL_PATH)
    channels = pd.read_csv(TARGETS_PATH)["target_channels"].tolist()
    logger.info(f"Training data shape: {train_df.shape} | Eval data shape: {eval_df.shape}")

    # Feature pruning on FULL training set
    input_feats = prune_correlated_features(train_df, channels, thresh=0.95)
    json.dump(input_feats, open(FEATURES_PATH, "w"), indent=2)
    logger.info(f"Using {len(input_feats)} input features.")

    # Prepare matrices
    X_train = train_df[input_feats].values
    y_train = train_df["is_anomaly"].values.astype(int)
    X_eval  = eval_df[input_feats].values
    y_eval  = eval_df["is_anomaly"].values.astype(int)

    # Train RF with the known best params
    logger.info("Training RandomForest with provided best hyperparameters...")
    model = RandomForestClassifier(**BEST_PARAMS)
    model.fit(X_train, y_train)

    # Save model params
    with open(MODEL_OUTPUT_PATH, "w") as f:
        json.dump(model.get_params(), f, indent=2)

    # Evaluation
    logger.info("Evaluating on EVAL...")
    y_probs_eval = model.predict_proba(X_eval)[:, 1] if len(np.unique(y_eval)) > 0 else np.zeros_like(y_eval, dtype=float)
    metrics_eval, y_pred_eval = evaluate_model_comprehensive(y_eval, y_probs_eval, beta=0.5, threshold=THRESHOLD)
    with open(RESULTS_EVAL_PATH, "w") as f:
        json.dump(metrics_eval, f, indent=2)
    np.save(CONFUSION_MATRIX_EVAL, np.array(metrics_eval["confusion_matrix"], dtype=int))

    # Evaluate on TRAIN (to build the train plot too)
    logger.info("Evaluating on TRAIN...")
    y_probs_train = model.predict_proba(X_train)[:, 1] if len(np.unique(y_train)) > 0 else np.zeros_like(y_train, dtype=float)
    metrics_train, y_pred_train = evaluate_model_comprehensive(y_train, y_probs_train, beta=0.5, threshold=THRESHOLD)
    with open(RESULTS_TRAIN_PATH, "w") as f:
        json.dump(metrics_train, f, indent=2)
    np.save(CONFUSION_MATRIX_TRAIN, np.array(metrics_train["confusion_matrix"], dtype=int))

    # Build events & plots
    logger.info("Building events and generating plots...")

    events_eval = build_events(eval_df, y_pred_eval)
    with open(EVENTS_EVAL_JSON, "w") as f:
        json.dump({"split": "eval", "threshold": THRESHOLD, "events": events_eval}, f, indent=2)
    plot_detected_vs_missed(events_eval, "RF: Detected vs Missed Events (Eval)", PLOT_COUNTS_EVAL)

    events_train = build_events(train_df, y_pred_train)
    with open(EVENTS_TRAIN_JSON, "w") as f:
        json.dump({"split": "train", "threshold": THRESHOLD, "events": events_train}, f, indent=2)
    plot_detected_vs_missed(events_train, "RF: Detected vs Missed Events (Train)", PLOT_COUNTS_TRAIN)

    logger.info("✅ Done. Saved:")
    logger.info(f"  - Params: {MODEL_OUTPUT_PATH}")
    logger.info(f"  - Features: {FEATURES_PATH}")
    logger.info(f"  - Results (eval/train): {RESULTS_EVAL_PATH} | {RESULTS_TRAIN_PATH}")
    logger.info(f"  - Confusion matrices: {CONFUSION_MATRIX_EVAL} | {CONFUSION_MATRIX_TRAIN}")
    logger.info(f"  - Events JSON: {EVENTS_EVAL_JSON} | {EVENTS_TRAIN_JSON}")
    logger.info(f"  - Plots: {PLOT_COUNTS_EVAL} | {PLOT_COUNTS_TRAIN}")
    logger.info(f"  - Logs: {LOG_FILE_PATH}")
    logger.info(f"Run dir: {RUN_DIR}")

if __name__ == "__main__":
    main()
