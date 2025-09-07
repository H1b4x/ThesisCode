#!/usr/bin/env python3
import pandas as pd
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
import resource
import sys
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
import pyarrow.parquet as pq

# Memory limiter
def limit_memory(max_gb: float):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    max_bytes = int(max_gb * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, hard))

try:
    limit_memory(85) #change the number 85 to allow more or less memory
except Exception as e:
    print(f"WARNING: Could not set memory limit: {e}", file=sys.stderr)

# Paths
SEED = 42
BASE_DIR = Path("/home/hajjohn1/bachelors-thesis")
TRAIN_PATH = BASE_DIR / "data" / "trainrf_lag.parquet"
EVAL_PATH  = BASE_DIR / "data" / "evalrf_lag.parquet"
RF_DIR = BASE_DIR / "final" / "RF" / "lag2event2"
RF_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RF_DIR / f"run_{timestamp}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE            = RUN_DIR / "training.log"
RESULTS_JSON        = RUN_DIR / "results.json"
CONF_MAT_NPY        = RUN_DIR / "confusion_matrix.npy"

#Eval outputs
EVENT_DET_JSON      = RUN_DIR / "events_detection.json"
PLOT_PROBS          = RUN_DIR / "rf_probabilities_timeline.png"
PLOT_BAR            = RUN_DIR / "rf_event_detection_bar.png"
PLOT_LEN_RATE       = RUN_DIR / "rf_event_detection_length_rate.png"
PLOT_LEN_COUNTS     = RUN_DIR / "rf_event_detection_length_counts.png"

# Train outputs
EVENT_DET_TRAIN_JSON   = RUN_DIR / "events_detection_train.json"
PLOT_PROBS_TRAIN       = RUN_DIR / "rf_probabilities_timeline_train.png"
PLOT_BAR_TRAIN         = RUN_DIR / "rf_event_detection_bar_train.png"
PLOT_LEN_RATE_TRAIN    = RUN_DIR / "rf_event_detection_length_rate_train.png"
PLOT_LEN_COUNTS_TRAIN  = RUN_DIR / "rf_event_detection_length_counts_train.png"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

#Helper functions

def read_parquet_in_chunks(file_path, chunk_size=None):
    dataset = pq.ParquetDataset(str(file_path))
    table = dataset.read()
    return table.to_pandas()


# Custom score function
def f_beta_corrected_stream(y_true, y_pred, beta=0.5):
    TP = FP = FN = N_neg = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0:
            N_neg += 1
        if yp == 1:
            if yt == 1:
                TP += 1
            else:
                FP += 1
        elif yt == 1:
            FN += 1
    precision_corr = 0.0
    if TP + FP > 0 and N_neg > 0:
        precision_corr = (TP / (TP + FP)) * (1 - FP / N_neg)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    b2 = beta * beta
    f_beta = ((1 + b2) * precision_corr * recall /
              (b2 * precision_corr + recall)) if (precision_corr + recall) > 0 else 0.0
    return f_beta, precision_corr, recall

def evaluate_model_comprehensive(y_true, y_probs, beta=0.5):
    y_pred = (y_probs > 0.5).astype(int)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    auc       = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0
    cm        = confusion_matrix(y_true, y_pred).tolist()
    f_beta, p_corr, r_corr = f_beta_corrected_stream(y_true, y_pred, beta)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "f_beta_corrected": f_beta,
        "precision_corrected": p_corr,
        "recall_corrected": r_corr,
        "confusion_matrix": cm,
        "n_samples": len(y_true)
    }

def build_events_from_binary(y_true_bin, y_pred_bin):
    events = []
    n = len(y_true_bin)
    i = 0
    eid = 0
    while i < n:
        if y_true_bin[i] == 1:
            start = i
            while i + 1 < n and y_true_bin[i + 1] == 1:
                i += 1
            end = i
            eid += 1
            detected = bool(np.any(y_pred_bin[start:end+1] == 1))
            events.append({
                "event_id": int(eid),
                "length": int(end - start + 1),
                "detected": detected
            })
        i += 1
    return events

# Minimal-leakage feature filter
EXCLUDE_EXACT = {"is_anomaly", "event_id"}  # hard blacklist
EXCLUDE_PATTERNS = ("anomaly", "label", "target", "future", "gt", "truth", "_y", "y_")

def looks_leaky(col: str) -> bool:
    c = col.lower()
    return (col in EXCLUDE_EXACT) or any(p in c for p in EXCLUDE_PATTERNS)

# Main

def main():
    np.random.seed(SEED)
    
    # Load raw data
    logger.info("Loading data…")
    train_raw = read_parquet_in_chunks(TRAIN_PATH)
    eval_raw  = read_parquet_in_chunks(EVAL_PATH)
    logger.info(f"  train: {train_raw.shape}, eval: {eval_raw.shape}")

    # Labels from raw frames
    y_train = train_raw["is_anomaly"].values
    y_eval  = eval_raw["is_anomaly"].values

    plot_train = train_raw[["is_anomaly"]].copy()
    plot_eval  = eval_raw[["is_anomaly"]].copy()

    # Reconstruct event_id on PLOT_EVAL if missing
    if "event_id" in eval_raw.columns and eval_raw["event_id"].notna().any():
        plot_eval["event_id"] = eval_raw["event_id"]
    else:
        mask = plot_eval["is_anomaly"] == 1
        event_start = mask & (~mask.shift(fill_value=False))
        event_ids = event_start.cumsum()
        plot_eval["event_id"] = np.where(mask, event_ids, np.nan)
        logger.info("Constructed `event_id` for EVAL (plot copy) from `is_anomaly`.")

    # Reconstruct event_id on PLOT_TRAIN if missing
    if "event_id" in train_raw.columns and train_raw["event_id"].notna().any():
        plot_train["event_id"] = train_raw["event_id"]
    else:
        mask_tr = plot_train["is_anomaly"] == 1
        event_start_tr = mask_tr & (~mask_tr.shift(fill_value=False))
        event_ids_tr = event_start_tr.cumsum()
        plot_train["event_id"] = np.where(mask_tr, event_ids_tr, np.nan)
        logger.info("Constructed `event_id` for TRAIN (plot copy) from `is_anomaly`.")

    # Feature selection
    # Use only columns present in BOTH splits, minus leaky ones
    shared_cols = set(train_raw.columns) & set(eval_raw.columns)
    features = sorted([c for c in shared_cols if not looks_leaky(c)])

    logger.info(f"Using {len(features)} features (leakage-filtered).")
    if "event_id" in features or "is_anomaly" in features:
        raise RuntimeError("Leakage filter failed: label-derived columns still present in features.")

    X_train = train_raw[features].astype(np.float32).values
    X_eval  = eval_raw[features].astype(np.float32).values

    # Best hyperparameters
    best_params = {
        "n_estimators": 103,
        "max_depth": 13,
        "min_samples_split": 5,
        "min_samples_leaf": 3,
        "max_features": None,
        "random_state": SEED,
        "n_jobs": -1
    }

    # Train final model
    logger.info("Training final RandomForest…")
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)

    # Predict on evaluation set
    y_probs = model.predict_proba(X_eval)[:, 1]
    metrics = evaluate_model_comprehensive(y_eval, y_probs)

    # Save metrics and confusion matrix
    logger.info("Saving results…")
    with open(RESULTS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(CONF_MAT_NPY, np.array(metrics["confusion_matrix"]))


    # Event-level detection with lengths (EVAL)
    events = []
    if "event_id" in plot_eval.columns:
        ev_df = plot_eval[["event_id", "is_anomaly"]].copy()
        ev_df["pred"] = (y_probs > 0.5).astype(int)
        for eid, grp in ev_df.groupby("event_id"):
            if isinstance(eid, float) and np.isnan(eid):
                continue
            if grp["is_anomaly"].any():
                length = int(grp["is_anomaly"].sum())
                detected = bool(grp["pred"].any())
                events.append({"event_id": int(eid), "length": length, "detected": detected})
        with open(EVENT_DET_JSON, "w") as f:
            json.dump({"split": "eval", "events": events}, f, indent=2)
        total = len(events)
        det   = sum(e["detected"] for e in events)
        if total > 0:
            logger.info(f"[EVAL] Event detection: {det}/{total} ({det/total*100:.1f}%)")
        else:
            logger.warning("[EVAL] No true events found for event-level analysis.")

        # Prepare DataFrame for plotting (EVAL)
        df_events = pd.DataFrame(events)
        if len(df_events) > 0:
            max_len = int(df_events['length'].max())
            bins    = [1, 1_000, 10_000, 20_000, 30_000, 60_000, max_len + 1]
            labels  = ["1–1k","1k–10k","10k–20k","20k–30k","30k–60k", f"60k–{max_len}"]
            df_events['length_bin'] = pd.cut(
                df_events['length'], bins=bins, labels=labels,
                right=False, include_lowest=True
            )
            grouped = (
                df_events.groupby('length_bin')['detected']
                         .agg(total_events='count', detected_events='sum')
            )
            grouped['detection_rate'] = grouped['detected_events'] / grouped['total_events'] * 100

            # Plot detection rate by event length (EVAL)
            plt.figure(figsize=(8,4))
            plt.bar(grouped.index.astype(str), grouped['detection_rate'])
            plt.xlabel('Event Length Range')
            plt.ylabel('Detection Rate (%)')
            plt.title('RF Detection Rate by Event Length (Eval)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(PLOT_LEN_RATE, dpi=150)
            plt.close()

            # Plot counts detected vs missed by event length (EVAL)
            undetected = grouped['total_events'] - grouped['detected_events']
            x = np.arange(len(grouped.index))
            width = 0.4
            plt.figure(figsize=(8,4))
            plt.bar(x-width/2, grouped['detected_events'], width, label='Detected')
            plt.bar(x+width/2, undetected, width, label='Missed')
            plt.xticks(x, grouped.index.astype(str), rotation=45, ha='right')
            plt.xlabel('Event Length Range')
            plt.ylabel('Number of Events')
            plt.title('RF Detected vs. Missed by Event Length (Eval)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(PLOT_LEN_COUNTS, dpi=150)
            plt.close()
        else:
            logger.warning("[EVAL] No true events found; skipping length-based plots.")
    else:
        logger.warning("No `event_id` column found in plot_eval; skipping per-event output and length-based plots")


    # Event-level detection with lengths (TRAIN)
    logger.info("Evaluating detection on TRAIN set for event breakdowns and plots…")
    y_probs_train = model.predict_proba(X_train)[:, 1]
    y_pred_train  = (y_probs_train > 0.5).astype(int)

    events_train = []
    if "event_id" in plot_train.columns:
        tr_df = plot_train[["event_id", "is_anomaly"]].copy()
        tr_df["pred"] = y_pred_train
        for eid, grp in tr_df.groupby("event_id"):
            if isinstance(eid, float) and np.isnan(eid):
                continue
            if grp["is_anomaly"].any():
                length = int(grp["is_anomaly"].sum())
                detected = bool(grp["pred"].any())
                events_train.append({"event_id": int(eid), "length": length, "detected": detected})
        with open(EVENT_DET_TRAIN_JSON, "w") as f:
            json.dump({"split": "train", "events": events_train}, f, indent=2)
        total_tr = len(events_train)
        det_tr   = sum(e["detected"] for e in events_train)
        if total_tr > 0:
            logger.info(f"[TRAIN] Event detection: {det_tr}/{total_tr} ({det_tr/total_tr*100:.1f}%)")
        else:
            logger.info("[TRAIN] No true events found.")

        # Prepare DataFrame for plotting (TRAIN)
        df_events_tr = pd.DataFrame(events_train)
        if len(df_events_tr) > 0:
            max_len_tr = int(df_events_tr['length'].max())
            bins_tr    = [1, 1_000, 10_000, 20_000, 30_000, 60_000, max_len_tr + 1]
            labels_tr  = ["1–1k","1k–10k","10k–20k","20k–30k","30k–60k", f"60k–{max_len_tr}"]
            df_events_tr['length_bin'] = pd.cut(
                df_events_tr['length'], bins=bins_tr, labels=labels_tr,
                right=False, include_lowest=True
            )
            grouped_tr = (
                df_events_tr.groupby('length_bin')['detected']
                            .agg(total_events='count', detected_events='sum')
            )
            grouped_tr['detection_rate'] = grouped_tr['detected_events'] / grouped_tr['total_events'] * 100

            # Plot detection rate by event length (TRAIN)
            plt.figure(figsize=(8,4))
            plt.bar(grouped_tr.index.astype(str), grouped_tr['detection_rate'])
            plt.xlabel('Event Length Range')
            plt.ylabel('Detection Rate (%)')
            plt.title('RF Detection Rate by Event Length (Train)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(PLOT_LEN_RATE_TRAIN, dpi=150)
            plt.close()

            # Plot counts detected vs missed by event length (TRAIN)
            undetected_tr = grouped_tr['total_events'] - grouped_tr['detected_events']
            x_tr = np.arange(len(grouped_tr.index))
            width_tr = 0.4
            plt.figure(figsize=(8,4))
            plt.bar(x_tr-width_tr/2, grouped_tr['detected_events'], width_tr, label='Detected')
            plt.bar(x_tr+width_tr/2, undetected_tr, width_tr, label='Missed')
            plt.xticks(x_tr, grouped_tr.index.astype(str), rotation=45, ha='right')
            plt.xlabel('Event Length Range')
            plt.ylabel('Number of Events')
            plt.title('RF Detected vs. Missed by Event Length (Train)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(PLOT_LEN_COUNTS_TRAIN, dpi=150)
            plt.close()
        else:
            logger.warning("[TRAIN] No true events found; skipping length-based plots.")
    else:
        logger.warning("No `event_id` column found in plot_train; skipping per-event output and length-based plots")

    # Plots
    logger.info("Generating timeline and summary bar plots…")
    idx = np.arange(len(y_probs))

    # Probability timeline (EVAL)
    plt.figure(figsize=(10,4))
    plt.plot(idx, y_probs, label="RF P(anomaly)")
    plt.scatter(idx[y_eval==1], y_probs[y_eval==1], marker="x", label="True anomaly")
    plt.scatter(idx[(y_probs>0.5)&(y_eval==0)],
                y_probs[(y_probs>0.5)&(y_eval==0)],
                facecolors="none", edgecolors="r", label="False positive")
    plt.axhline(0.5, linestyle="--", label="Threshold")
    plt.title("Random Forest — Anomaly Probabilities on Evaluation Set")
    plt.xlabel("Sample Index")
    plt.ylabel("Probability of Anomaly")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOT_PROBS, dpi=150)
    plt.close()

    # Probability timeline (TRAIN)
    idx_tr = np.arange(len(y_probs_train))
    plt.figure(figsize=(10,4))
    plt.plot(idx_tr, y_probs_train, label="RF P(anomaly)")
    plt.scatter(idx_tr[y_train==1], y_probs_train[y_train==1], marker="x", label="True anomaly")
    plt.scatter(idx_tr[(y_probs_train>0.5)&(y_train==0)],
                y_probs_train[(y_probs_train>0.5)&(y_train==0)],
                facecolors="none", edgecolors="r", label="False positive")
    plt.axhline(0.5, linestyle="--", label="Threshold")
    plt.title("Random Forest — Anomaly Probabilities on Training Set")
    plt.xlabel("Sample Index")
    plt.ylabel("Probability of Anomaly")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOT_PROBS_TRAIN, dpi=150)
    plt.close()

    # Event detection bar (overall counts) — EVAL
    if "event_id" in plot_eval.columns:
        if not events:
            logger.warning("No true events were found for EVAL bar plot.")
        else:
            det_count = sum(e["detected"] for e in events)
            und_count = len(events) - det_count
            plt.figure(figsize=(4,4))
            plt.bar(["Detected","Undetected"], [det_count, und_count])
            plt.title("RF with Lag Features: Detected vs. Missed Events (Eval)")
            plt.ylabel("Number of True Events")
            plt.tight_layout()
            plt.savefig(PLOT_BAR, dpi=150)
            plt.close()

    # Event detection bar (overall counts) — TRAIN
    if "event_id" in plot_train.columns:
        if not events_train:
            logger.warning("No true events were found for TRAIN bar plot.")
        else:
            det_count_tr = sum(e["detected"] for e in events_train)
            und_count_tr = len(events_train) - det_count_tr
            plt.figure(figsize=(4,4))
            plt.bar(["Detected","Undetected"], [det_count_tr, und_count_tr])
            plt.title("RF with Lag Features: Detected vs. Missed Events (Train)")
            plt.ylabel("Number of True Events")
            plt.tight_layout()
            plt.savefig(PLOT_BAR_TRAIN, dpi=150)
            plt.close()

    logger.info(f"Artifacts written to {RUN_DIR}")

if __name__ == "__main__":
    main()
