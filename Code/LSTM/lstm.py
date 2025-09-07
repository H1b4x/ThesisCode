import os
import torch
import torch.nn as nn
import optuna
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
from pathlib import Path
from datetime import datetime
import logging
import json
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
N_TRIALS = 50
TIMEOUT_SECONDS = 60 * 60 * 5  # 5 hours

# Paths
BASE_DIR = Path("/home/hajjohn1/bachelors-thesis")
TRAIN_PATH = BASE_DIR / "data" / "training.parquet"
EVAL_PATH = BASE_DIR / "data" / "evaluation.parquet"
TARGETS_PATH = BASE_DIR / "data" / "target_channels.csv"
LSTM_DIR = BASE_DIR / "final" / "LSTM" / "lstm2"
LSTM_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = LSTM_DIR / f"run_{timestamp}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_PATH       = RUN_DIR / "lstm_model_state_dict.pt"
LOG_FILE_PATH           = RUN_DIR / "training.log"
RESULTS_PATH            = RUN_DIR / "results.json"
HYPERPARAMS_PATH        = RUN_DIR / "best_hyperparameters.json"
CONFUSION_MATRIX_PATH   = RUN_DIR / "confusion_matrix.npy"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load data
logger.info("Loading data...")
train_df = pd.read_parquet(TRAIN_PATH)
eval_df  = pd.read_parquet(EVAL_PATH)
channels = pd.read_csv(TARGETS_PATH)["target_channels"].tolist()

# Feature pruning
def prune_correlated_features(df, feats, thresh=0.95):
    logger.info(f"Pruning features with correlation > {thresh} over full dataset...")
    corr = df[feats].corr().abs()
    mask = np.triu(np.ones_like(corr), k=1).astype(bool)
    high_corr = corr.where(mask).stack().loc[lambda x: x > thresh].index
    to_drop = set(f2 for f1, f2 in high_corr)
    pruned = [f for f in feats if f not in to_drop]
    logger.info(f"Pruned {len(feats) - len(pruned)} features.")
    return pruned

input_feats = prune_correlated_features(train_df, channels, thresh=0.95)

# Dataset
class WindowDataset(Dataset):
    def __init__(self, X, y, window_size, starts):
        self.X = X
        self.y = y
        self.ws = window_size
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        return self.X[s:s+self.ws], self.y[s:s+self.ws]

# Model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.out(out)).squeeze(-1)

# Custom score function and evaluation
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
    f_beta = ((1 + b2) * precision_corr * recall / (b2 * precision_corr + recall)) \
        if (precision_corr + recall) > 0 else 0.0
    return f_beta, precision_corr, recall

def evaluate_seq_predictions(y_true_full, pred_windows, starts, window_size, threshold):
    total_len = len(y_true_full)
    votes  = np.zeros(total_len)
    counts = np.zeros(total_len)

    for pred, start in zip(pred_windows, starts):
        end = start + window_size
        if end > total_len:
            continue
        votes[start:end] += pred
        counts[start:end] += 1

    averaged = np.divide(votes, counts, out=np.zeros_like(votes), where=counts > 0)
    binary  = (averaged > threshold).astype(int)

    f_beta, precision_corr, recall = f_beta_corrected_stream(y_true_full, binary, beta=0.5)
    auc = roc_auc_score(y_true_full, averaged) if len(np.unique(y_true_full)) > 1 else 0.0
    cm  = confusion_matrix(y_true_full, binary).tolist()

    return {
        'f0.5': f_beta,
        'precision_corr': precision_corr,
        'recall': recall,
        'auc': auc,
        'confusion_matrix': cm,
        'n_samples': total_len
    }

# Optuna objective
def objective(trial):
    window_size = trial.suggest_categorical("window_size", [256, 512, 1024, 2048])
    stride      = trial.suggest_int("stride", window_size // 8, window_size // 2)
    hidden_size = trial.suggest_int("hidden_size", 32, 256)
    num_layers  = trial.suggest_int("num_layers", 1, 3)
    dropout     = trial.suggest_float("dropout", 0.0, 0.5)
    lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size  = trial.suggest_categorical("batch_size", [64, 128, 256])

    # half-data for speed
    half_idx = len(train_df) // 2
    trial_df = train_df.iloc[half_idx:].reset_index(drop=True)
    X = torch.from_numpy(trial_df[input_feats].values.astype(np.float32))
    y = torch.from_numpy(trial_df["is_anomaly"].values.astype(np.float32))

    # build windows
    if window_size > X.shape[0]:
        raise ValueError(f"window_size ({window_size}) > data length ({X.shape[0]})")
    max_stride = X.shape[0] - window_size
    stride = min(stride, max(1, max_stride))
    starts = torch.arange(0, X.shape[0] - window_size, step=stride)
    split = int(0.8 * len(starts))
    train_starts, val_starts = starts[:split], starts[split:]

    train_dl = DataLoader(WindowDataset(X, y, window_size, train_starts),
                          batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(WindowDataset(X, y, window_size, val_starts),
                        batch_size=batch_size, shuffle=False)

    # quick train
    model = LSTMClassifier(len(input_feats), hidden_size, num_layers, dropout).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    for _ in range(5):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = loss_fn(model(xb), yb)
            optim.zero_grad()
            loss.backward()
            optim.step()

    # validation predictions
    pred_windows = []
    model.eval()
    with torch.no_grad():
        for xb, _ in val_dl:
            xb = xb.to(DEVICE)
            pred_windows.append(model(xb).cpu().numpy())
    pred_windows = np.vstack(pred_windows)

    # evaluate at threshold=0.9
    metrics = evaluate_seq_predictions(
        y_true_full=y.numpy(),
        pred_windows=pred_windows,
        starts=val_starts.numpy(),
        window_size=window_size,
        threshold=0.9
    )
    return metrics['f0.5']

# Run hyperparameter search
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS)

best = study.best_trial.params
logger.info(f"Best hyperparameters:\n{json.dumps(best, indent=2)}")

# Final training on full data
X_full = torch.from_numpy(train_df[input_feats].values.astype(np.float32))
y_full = torch.from_numpy(train_df["is_anomaly"].values.astype(np.float32))

stride      = best["stride"]
window_size = best["window_size"]
max_stride  = X_full.shape[0] - window_size
if stride > max_stride:
    stride = max(1, max_stride)

starts_full = torch.arange(0, X_full.shape[0] - window_size, step=stride)
train_dl_full = DataLoader(WindowDataset(X_full, y_full, window_size, starts_full),
                           batch_size=best["batch_size"], shuffle=True)

model = LSTMClassifier(len(input_feats),
                       best["hidden_size"],
                       best["num_layers"],
                       best["dropout"]).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=best["lr"])
loss_fn = nn.BCELoss()

logger.info("Starting full training...")
for epoch in range(10):
    model.train()
    total_loss = 0.0
    for xb, yb in train_dl_full:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(starts_full)
    logger.info(f"Epoch {epoch+1}/10 | Loss: {avg_loss:.4f}")

# Eval set predictions
eval_X = torch.from_numpy(eval_df[input_feats].values.astype(np.float32))
eval_y = torch.from_numpy(eval_df["is_anomaly"].values.astype(np.float32))
eval_starts = torch.arange(0, eval_X.shape[0] - window_size, step=stride)
eval_dl = DataLoader(WindowDataset(eval_X, eval_y, window_size, eval_starts),
                     batch_size=best["batch_size"], shuffle=False)

model.eval()
pred_windows = []
with torch.no_grad():
    for xb, _ in eval_dl:
        xb = xb.to(DEVICE)
        pred_windows.append(model(xb).cpu().numpy())
pred_windows = np.vstack(pred_windows) if pred_windows else np.empty((0, window_size))

# Second Optuna study: threshold tuning on eval set
def threshold_objective(trial):
    t = trial.suggest_float("threshold", 0.85, 0.95)
    metrics_t = evaluate_seq_predictions(
        y_true_full=eval_y.numpy(),
        pred_windows=pred_windows,
        starts=eval_starts.numpy(),
        window_size=window_size,
        threshold=t
    )
    return metrics_t["f0.5"]

thresh_study = optuna.create_study(direction="maximize")
thresh_study.optimize(threshold_objective, n_trials=10)

best_threshold = thresh_study.best_trial.params["threshold"]
best["threshold"] = best_threshold
logger.info(f"Selected threshold: {best_threshold:.4f} with F0.5={thresh_study.best_value:.4f}")

# Final metrics
final_metrics = evaluate_seq_predictions(
    y_true_full=eval_y.numpy(),
    pred_windows=pred_windows,
    starts=eval_starts.numpy(),
    window_size=window_size,
    threshold=best_threshold
)

# Save artifacts
torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
with open(HYPERPARAMS_PATH, 'w') as f:
    json.dump(best, f, indent=2)
with open(RESULTS_PATH, 'w') as f:
    json.dump(final_metrics, f, indent=2)
np.save(CONFUSION_MATRIX_PATH, np.array(final_metrics["confusion_matrix"]))

logger.info(f"✅ Training complete. Model, threshold, and results saved in: {RUN_DIR}")
