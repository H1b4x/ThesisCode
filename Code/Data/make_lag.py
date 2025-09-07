#This script makes the two lag feature sets, 
#Only need to change two lines to make the different sets
#loads data, keeps relevent features, prunes highly correlated features, builds lags on the two datasets separately
# and removes NaN rows


from pathlib import Path
import json

import numpy as np
import pandas as pd


TRAIN_PATH   = "/home/hajjohn1/bachelors-thesis/data/training.parquet"
EVAL_PATH    = "/home/hajjohn1/bachelors-thesis/data/evaluation.parquet"
TARGETS_PATH = "/home/hajjohn1/bachelors-thesis/data/target_channels.csv"
OUT_DIR      = "/home/hajjohn1/bachelors-thesis/data"  

TRAIN_LAG_OUT = Path(OUT_DIR) / "traintest_lag.parquet"
EVAL_LAG_OUT  = Path(OUT_DIR) / "evaltest_lag.parquet"
DROPPED_JSON  = Path(OUT_DIR) / "droppedtest_channels.json"

ID_COL    = "id"
LABEL_COL = "is_anomaly"

# Lag settings 
LAGS = [1, 2, 4, 8, 32, 64, 128, 256, 1024, 2048, 4096, 8192] #larger set
#LAGS = [1, 2, 4, 8, 32, 64, 128, 256, 1024, 2048] # for random forest and isolation forest

# Correlation threshold
CORR_THRESH = 0.95 #0.995 for larger set


DROP_NANS = True

def add_lags(df: pd.DataFrame, channels, lags):
    for ch in channels:
        for lag in lags:
            df[f"{ch}_lag{lag}"] = df[ch].shift(lag)
    return df

def drop_high_corr(df: pd.DataFrame, features, thr: float):
    corr = df[features].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > thr)]
    return to_drop


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(TRAIN_PATH)
    eval_df  = pd.read_parquet(EVAL_PATH)
    target_channels = pd.read_csv(TARGETS_PATH)["target_channels"].tolist()

    cols = [ID_COL] + target_channels + [LABEL_COL]
    train_df = train_df[cols].sort_values(ID_COL).reset_index(drop=True)
    eval_df  = eval_df[cols].sort_values(ID_COL).reset_index(drop=True)

    dropped = drop_high_corr(train_df, target_channels, CORR_THRESH)
    kept_channels = [c for c in target_channels if c not in dropped]

    with DROPPED_JSON.open("w") as f:
        json.dump({"correlation_threshold": CORR_THRESH,
                   "dropped_channels": dropped,
                   "kept_channels": kept_channels}, f, indent=2)

    print(f"Channels before: {len(target_channels)} | dropped: {len(dropped)} | kept: {len(kept_channels)}")
    if dropped:
        print("Dropped due to high correlation:", dropped)


    train_df = train_df[[ID_COL] + kept_channels + [LABEL_COL]]
    eval_df  = eval_df[[ID_COL] + kept_channels + [LABEL_COL]]


    train_lag = add_lags(train_df.copy(), kept_channels, LAGS)
    eval_lag  = add_lags(eval_df.copy(),  kept_channels, LAGS)

    if DROP_NANS:
        train_lag = train_lag.dropna().reset_index(drop=True)
        eval_lag  = eval_lag.dropna().reset_index(drop=True)


    train_lag.to_parquet(TRAIN_LAG_OUT, index=False)
    eval_lag.to_parquet(EVAL_LAG_OUT,  index=False)

    print(f"Saved train lag set to: {TRAIN_LAG_OUT}  (rows={len(train_lag)}, cols={train_lag.shape[1]})")
    print(f"Saved eval  lag set to: {EVAL_LAG_OUT}   (rows={len(eval_lag)},  cols={eval_lag.shape[1]})")
    print(f"Saved dropped/kept channel list to: {DROPPED_JSON}")

if __name__ == "__main__":
    main()
