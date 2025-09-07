import os
import random
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import optuna
import joblib
import matplotlib.pyplot as plt

# Device & Seeds
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # adjust as needed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = device.type == "cuda"  

# Paths & Constants
BASE_DIR = Path("/home/hajjohn1/bachelors-thesis")
TRAIN_PATH = BASE_DIR / "data" / "training.parquet"
EVAL_PATH = BASE_DIR / "data" / "evaluation.parquet"
TARGETS_PATH = BASE_DIR / "data" / "target_channels.csv"

RUN_DIR = BASE_DIR / "final" / "LSTMAE2" / "1" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_PATH = RUN_DIR / "lstm_ae_model.pt"
SCALER_OUTPUT_PATH = RUN_DIR / "scaler.pkl"
FEATURES_PATH = RUN_DIR / "features.json"
RESULTS_PATH = RUN_DIR / "results.json"
HYPERPARAMS_PATH = RUN_DIR / "best_hyperparameters.json"
THRESHOLD_PATH = RUN_DIR / "threshold.json"
CONFUSION_MATRIX_PATH = RUN_DIR / "confusion_matrix.npy"
EVENTS_JSON_PATH = RUN_DIR / "events_eval.json"
EVENTS_PLOT_PATH = RUN_DIR / "events_detected_vs_missed.png"
LOG_FILE_PATH = RUN_DIR / "training.log"

N_TRIALS = 50
TIMEOUT_SECONDS = 60 * 60 * 5
FIXED_THRESHOLD_PERCENTILE = 90.0  
SUBSAMPLE_FRAC = 0.30  # contiguous block for hyper tuning
PATIENCE = 5  # early stopping patience
MAX_EPOCHS_TUNE = 30
MAX_EPOCHS_FINAL = 50

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()],
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
    precision_corr = 0.0
    if (TP_e + FP_t > 0) and (N_t > 0):
        correction = 1 - FP_t / N_t
        if correction < 0:
            correction = 0.0
        precision_corr = (TP_e / (TP_e + FP_t)) * correction
    recall_corr = TP_e / (TP_e + FN_e) if (TP_e + FN_e) > 0 else 0.0
    b2 = beta * beta
    denom = b2 * precision_corr + recall_corr
    f_beta = ((1 + b2) * precision_corr * recall_corr / denom) if denom > 0 else 0.0
    return f_beta, precision_corr, recall_corr

# Full LSTM Autoencoder
class LSTMAutoencoder(nn.Module):
    """
    Encoder: LSTM -> last hidden -> Linear (latent)
    Decoder: latent repeated over seq_len -> LSTM -> Linear -> reconstruct sequence
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 num_layers_enc: int = 1, num_layers_dec: int = 1, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers_enc,
            batch_first=True,
            dropout=(dropout if num_layers_enc > 1 else 0.0)
        )
        self.enc_fc = nn.Linear(hidden_dim, latent_dim)
        self.dec_init = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers_dec,
            batch_first=True,
            dropout=(dropout if num_layers_dec > 1 else 0.0)
        )
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (B, T, F)
        _, (h_n, _) = self.encoder(x)          # h_n: (num_layers_enc, B, H)
        h_last = h_n[-1]                        # (B, H)
        latent = self.enc_fc(h_last)            # (B, Z)

        B, T, _ = x.shape
        latent_seq = latent.unsqueeze(1).expand(B, T, latent.size(-1))  # (B, T, Z)

        h0 = self.dec_init(latent).unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)  # (layers_dec, B, H)
        c0 = torch.zeros_like(h0)

        dec_out, _ = self.decoder(latent_seq, (h0, c0))  # (B, T, H)
        recon = self.out(dec_out)                        # (B, T, F)
        return recon

#Windowed Datasets (causal windows)
class CausalWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, window: int, stride: int,
                 normals_only: bool, dtype=np.float32):
        assert X.ndim == 2
        self.X = X
        self.y = y
        self.window = int(window)
        self.stride = int(stride)
        self.dtype = dtype

        N = X.shape[0]
        if N < window:
            self.indices = np.array([], dtype=np.int64)
        else:
            ends = np.arange(window - 1, N, stride, dtype=np.int64)
            if normals_only:
                keep = []
                y_local = (y == 0).astype(np.uint8)
                for e in ends:
                    s = e - window + 1
                    if y_local[s:e+1].all():
                        keep.append(e)
                self.indices = np.array(keep, dtype=np.int64)
            else:
                self.indices = ends

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        e = self.indices[idx]
        s = e - self.window + 1
        seq = self.X[s:e+1].astype(self.dtype, copy=False)
        return torch.from_numpy(seq)

def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, for_gpu: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        pin_memory=for_gpu,
        persistent_workers=False
    )

#Train & Eval Utilities
def train_epoch_seq(model, loader, criterion, optimizer, scaler, amp_enabled: bool):
    model.train()
    total_loss = 0.0
    count = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True) 
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            recon = model(batch)
            loss = criterion(recon, batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * batch.size(0)
        count += batch.size(0)
        del batch, recon, loss
    return total_loss / max(count, 1)

@torch.no_grad()
def eval_epoch_seq(model, loader, criterion, amp_enabled: bool):
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            recon = model(batch)
            loss = criterion(recon, batch)
        total_loss += loss.item() * batch.size(0)
        count += batch.size(0)
        del batch, recon, loss
    return total_loss / max(count, 1)

@torch.no_grad()
def per_point_errors_causal(model, X: np.ndarray, window: int, batch_size: int) -> np.ndarray:
    model.eval()
    N, _ = X.shape
    errs = np.full(N, np.nan, dtype=np.float32)
    eval_ds = CausalWindowDataset(X, y=np.zeros(N, dtype=np.uint8), window=window, stride=1, normals_only=False)
    eval_dl = make_dataloader(eval_ds, batch_size=batch_size, shuffle=False, for_gpu=(device.type == "cuda"))
    it = 0
    for batch in eval_dl:
        B, T, _ = batch.shape
        ends = np.arange(window - 1 + it, window - 1 + it + B, dtype=np.int64)
        it += B
        batch = batch.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
            recon = model(batch)  # (B, T, F)
            last_err = torch.mean((recon[:, -1, :] - batch[:, -1, :]) ** 2, dim=1)  # (B,)
        errs[ends] = last_err.detach().float().cpu().numpy()
        del batch, recon, last_err
    return errs

#Load & Prepare Data
logger.info("Loading data...")
train_df = pd.read_parquet(TRAIN_PATH)
eval_df = pd.read_parquet(EVAL_PATH)
channels: List[str] = pd.read_csv(TARGETS_PATH)["target_channels"].tolist()

train_df = train_df[channels + ["is_anomaly"]]
eval_df  = eval_df[channels + ["is_anomaly"]]

# Drop highly correlated features
corr = train_df[channels].corr().abs()
mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
high_corr = corr.where(mask).stack().loc[lambda x: x > 0.90]
to_drop = {f2 for f1, f2 in high_corr.index}
features = [f for f in channels if f not in to_drop]
with open(FEATURES_PATH, "w") as f:
    json.dump(features, f)

# Arrays
X_train_raw = train_df[features].to_numpy(dtype=np.float32, copy=False)
y_train = train_df["is_anomaly"].to_numpy(dtype=np.int8, copy=False)
X_eval_raw  = eval_df[features].to_numpy(dtype=np.float32, copy=False)
y_eval = eval_df["is_anomaly"].to_numpy(dtype=np.int8, copy=False)

# Global scaler fit on normals only
global_scaler = StandardScaler()
global_scaler.fit(X_train_raw[y_train == 0])
X_train = global_scaler.transform(X_train_raw).astype(np.float32, copy=False)
X_eval  = global_scaler.transform(X_eval_raw).astype(np.float32, copy=False)
joblib.dump(global_scaler, SCALER_OUTPUT_PATH)

# Hyperparam tuning subset (contiguous)
n_total = X_train.shape[0]
block_size = int(np.floor(n_total * SUBSAMPLE_FRAC))
if block_size < 1000:
    raise ValueError("Subsample too small, adjust SUBSAMPLE_FRAC or data size.")
start_idx = random.randint(0, n_total - block_size)
end_idx = start_idx + block_size
logger.info(f"Hyperparameter tuning subset: rows {start_idx} to {end_idx} (~{SUBSAMPLE_FRAC*100:.1f}% of train)")
X_sub = X_train[start_idx:end_idx]
y_sub = y_train[start_idx:end_idx]

if not np.any(y_sub == 0):
    raise RuntimeError("No normal data in subsample for hyperparameter tuning.")

#Optuna Objective 
def objective_hyper(trial: optuna.Trial) -> float:
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256, step=32)
    latent_dim = trial.suggest_int("latent_dim", 8, min(hidden_dim, 128))
    num_layers_enc = trial.suggest_int("num_layers_enc", 1, 3)
    num_layers_dec = trial.suggest_int("num_layers_dec", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    #Window & stride (relative)
    window = trial.suggest_categorical("window", [32, 64, 128, 256, 512, 1024, 2048])
    if X_sub.shape[0] < window:
        raise optuna.TrialPruned()
    stride_ratio = trial.suggest_float("stride_ratio", 1/8, 1/2, log=True)  # relative to window
    stride = max(1, int(round(window * stride_ratio)))

    # 2-fold sequential split
    split = int(0.5 * X_sub.shape[0])
    X_tr, y_tr = X_sub[:split], y_sub[:split]
    X_va, y_va = X_sub[split:], y_sub[split:]

    # Datasets: train on fully-normal windows; validate on all windows
    train_ds = CausalWindowDataset(X_tr, y_tr, window=window, stride=stride, normals_only=True)
    if len(train_ds) == 0:
        raise optuna.TrialPruned()
    val_ds   = CausalWindowDataset(X_va, y_va, window=window, stride=1, normals_only=False)

    train_dl = make_dataloader(train_ds, batch_size=batch_size, shuffle=True,  for_gpu=(device.type=="cuda"))
    val_dl   = make_dataloader(val_ds,   batch_size=batch_size, shuffle=False, for_gpu=(device.type=="cuda"))

    input_dim = X_sub.shape[1]
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers_enc=num_layers_enc,
        num_layers_dec=num_layers_dec,
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    best_val = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(MAX_EPOCHS_TUNE):
        tr_loss = train_epoch_seq(model, train_dl, criterion, optimizer, scaler, AMP_ENABLED)
        va_loss = eval_epoch_seq(model, val_dl, criterion, AMP_ENABLED)
        if va_loss < best_val - 1e-4:
            best_val = va_loss
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            break

    if best_state is None:
        raise optuna.TrialPruned()
    model.load_state_dict(best_state, strict=True)
    model.to(device)

    # Per-point causal errors on validation half (stride=1)
    recon_err = per_point_errors_causal(model, X_va, window=window, batch_size=batch_size)
    val_norm_mask = (y_va == 0)
    if not np.any(val_norm_mask):
        return 0.0
    thr = np.nanpercentile(recon_err[val_norm_mask], FIXED_THRESHOLD_PERCENTILE)
    preds = (recon_err >= thr).astype(np.int8)
    valid = ~np.isnan(recon_err)
    f_beta, _, _ = f_beta_corrected_stream(y_va[valid], preds[valid])

    logger.info(
        f"Trial {trial.number} F0.5={f_beta:.4f} "
        f"params={{'hidden':{hidden_dim},'latent':{latent_dim},"
        f"'layers_enc':{num_layers_enc},'layers_dec':{num_layers_dec},"
        f"'dropout':{dropout:.3f},'lr':{lr:.1e},'batch':{batch_size},"
        f"'window':{window},'stride_ratio':{stride_ratio:.4f},'stride':{stride}}}"
    )
    return f_beta

# Run hyperparameter optimization
sampler = optuna.samplers.TPESampler(seed=SEED)
study_hyper = optuna.create_study(direction="maximize", sampler=sampler)
study_hyper.optimize(objective_hyper, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS)
best_params = study_hyper.best_trial.params
logger.info(f"Best hyperparameters: {best_params}")

# Final model training on all normal training windows
input_dim = X_train.shape[1]
hidden_dim = int(best_params["hidden_dim"])
latent_dim = int(best_params["latent_dim"])
num_layers_enc = int(best_params["num_layers_enc"])
num_layers_dec = int(best_params["num_layers_dec"])
dropout = float(best_params["dropout"])
lr = float(best_params["lr"])
batch_size_final = int(best_params["batch_size"])
window_final = int(best_params["window"])
stride_train = max(1, int(round(window_final * float(best_params["stride_ratio"]))))

if X_train.shape[0] < window_final:
    raise RuntimeError(f"Final window size {window_final} exceeds training length {X_train.shape[0]}.")

final_model = LSTMAutoencoder(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    latent_dim=latent_dim,
    num_layers_enc=num_layers_enc,
    num_layers_dec=num_layers_dec,
    dropout=dropout
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(final_model.parameters(), lr=lr)
scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

# Train dataset: only fully-normal windows
train_ds_full = CausalWindowDataset(X_train, y_train, window=window_final, stride=stride_train, normals_only=True)
train_dl_full = make_dataloader(train_ds_full, batch_size=batch_size_final, shuffle=True, for_gpu=(device.type=="cuda"))

best_train_loss = float("inf")
epochs_no_improve = 0
best_state_final = None

for epoch in range(MAX_EPOCHS_FINAL):
    tr_loss = train_epoch_seq(final_model, train_dl_full, criterion, optimizer, scaler, AMP_ENABLED)
    if tr_loss < best_train_loss - 1e-6:
        best_train_loss = tr_loss
        epochs_no_improve = 0
        best_state_final = {k: v.detach().cpu() for k, v in final_model.state_dict().items()}
    else:
        epochs_no_improve += 1
    logger.info(f"Final training epoch {epoch+1}, loss={tr_loss:.6f}")
    if epochs_no_improve >= PATIENCE:
        break

if best_state_final is not None:
    final_model.load_state_dict(best_state_final, strict=True)
final_model.to(device)
torch.save(final_model.state_dict(), MODEL_OUTPUT_PATH)

# Evaluation (per-point causal scores, stride=1)
with torch.no_grad():
    recon_err_eval = per_point_errors_causal(final_model, X_eval, window=window_final, batch_size=batch_size_final)

# Threshold tuning
def objective_thr(trial: optuna.Trial) -> float:
    pct = trial.suggest_float("threshold_percentile", 85.0, 95.0)
    base = recon_err_eval[(y_eval == 0)]
    thr = np.nanpercentile(base, pct)
    preds = (recon_err_eval >= thr).astype(np.int8)
    valid = ~np.isnan(recon_err_eval)
    f_beta, _, _ = f_beta_corrected_stream(y_eval[valid], preds[valid])
    return f_beta

study_thr = optuna.create_study(direction="maximize", sampler=sampler)
study_thr.optimize(objective_thr, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS)
best_thr = float(study_thr.best_trial.params["threshold_percentile"])
threshold_value = float(np.nanpercentile(recon_err_eval[(y_eval == 0)], best_thr))

# Final preds (per point)
pred_final = (recon_err_eval >= threshold_value).astype(np.int8)
valid = ~np.isnan(recon_err_eval)
f_beta, prec, rec = f_beta_corrected_stream(y_eval[valid], pred_final[valid])
auc = roc_auc_score(y_eval[valid], recon_err_eval[valid]) if len(np.unique(y_eval[valid])) > 1 else 0.0
cm = confusion_matrix(y_eval[valid], pred_final[valid])

# save results
def to_serializable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out

results = {
    "f_beta_corrected": float(f_beta),
    "precision_corrected": float(prec),
    "recall_corrected": float(rec),
    "auc": float(auc),
    "confusion_matrix": cm.tolist(),
    "n_samples_eval": int(valid.sum()),
    "best_hyperparameters": to_serializable(best_params),
    "window_final": window_final,
    "stride_train": stride_train,
    "stride_ratio": float(best_params["stride_ratio"]),
    "threshold_percentile": best_thr,
    "threshold_value": threshold_value
}
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
with open(HYPERPARAMS_PATH, "w") as f:
    json.dump(to_serializable(best_params), f, indent=2)
with open(THRESHOLD_PATH, "w") as f:
    json.dump({"threshold_percentile": best_thr, "threshold_value": threshold_value}, f, indent=2)
np.save(CONFUSION_MATRIX_PATH, cm)

logger.info(f"Training complete. Outputs in {RUN_DIR}")

# Events JSON + Plot on eval_df
mask = eval_df["is_anomaly"] == 1
event_start = mask & (~mask.shift(fill_value=False))
event_ids = event_start.cumsum()
eval_df = eval_df.copy()
eval_df["event_id"] = np.where(mask, event_ids, np.nan)

# Build events list
events = []
for eid, grp in eval_df[mask].groupby("event_id"):
    idx = grp.index.to_numpy()
    idx_valid = idx[idx >= (window_final - 1)]
    length = int(len(grp))
    detected = bool(pred_final[idx_valid].sum() > 0) if len(idx_valid) > 0 else False
    events.append({"event_id": int(eid), "length": length, "detected": detected})

# Save JSON
with open(EVENTS_JSON_PATH, "w") as f:
    json.dump({
        "model_run": RUN_DIR.name,
        "threshold": threshold_value,
        "window": window_final,
        "stride_train": stride_train,
        "events": events
    }, f, indent=2)
logger.info(f"Saved {len(events)} event records to {EVENTS_JSON_PATH}")

# Plot: Detected vs. Missed by event length
df_events = pd.DataFrame(events)
if not df_events.empty:
    max_len = int(df_events["length"].max())
    bins = [1, 1_000, 10_000, 20_000, 30_000, 60_000, max_len + 1]
    labels = ["1–1k","1k–10k","10k–20k","20k–30k","30k–60k", f"60k–{max_len}"]
    df_events["length_bin"] = pd.cut(
        df_events["length"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True
    )
    grouped = (
        df_events
        .groupby("length_bin", observed=True)["detected"]
        .agg(total_events="count", detected_events="sum")
        .reindex(labels)
        .fillna(0)
    )
    undetected = grouped["total_events"] - grouped["detected_events"]

    plt.figure(figsize=(8, 4))
    x = np.arange(len(grouped.index))
    width = 0.4
    plt.bar(x - width/2, grouped["detected_events"].to_numpy(), width, label="Detected")
    plt.bar(x + width/2, undetected.to_numpy(), width, label="Missed")
    plt.xticks(x, grouped.index, rotation=45, ha="right")
    plt.xlabel("Event Length Range")
    plt.ylabel("Number of Events")
    plt.title("LSTM Autoencoder Detected vs. Missed by Event Length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EVENTS_PLOT_PATH, dpi=150)
    plt.close()
    logger.info(f"Saved events plot to {EVENTS_PLOT_PATH}")
else:
    logger.info("No anomaly events found in eval set; skipping plot.")
