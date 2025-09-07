import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# Paths 
BASE_DIR       = Path("/home/hajjohn1/bachelors-thesis")
TRAIN_PATH     = BASE_DIR / "data" / "training.parquet"
EVAL_PATH      = BASE_DIR / "data" / "evaluation.parquet"
TARGETS_PATH   = BASE_DIR / "data" / "target_channels.csv"
OUTPUT_DIR     = BASE_DIR / "final" / "LSTM" / "fixed_run_event"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_STATE    = OUTPUT_DIR / "lstm_model_state_dict.pt"
RESULTS_PATH   = OUTPUT_DIR / "results.json"
EVENTS_PATH    = OUTPUT_DIR / "eval_event_detection_lstm_fixed_run.json"
EVENTS_TRAIN_PATH = OUTPUT_DIR / "train_event_detection_lstm_fixed_run.json"  # NEW

# New plot outputs
PLOT_LEN_RATE_EVAL   = OUTPUT_DIR / "lstm_event_detection_length_rate_eval.png"
PLOT_LEN_COUNTS_EVAL = OUTPUT_DIR / "lstm_event_detection_length_counts_eval.png"
PLOT_LEN_RATE_TRAIN  = OUTPUT_DIR / "lstm_event_detection_length_rate_train.png"
PLOT_LEN_COUNTS_TRAIN= OUTPUT_DIR / "lstm_event_detection_length_counts_train.png"
PLOT_PROBS_EVAL      = OUTPUT_DIR / "lstm_probabilities_timeline_eval.png"
PLOT_PROBS_TRAIN     = OUTPUT_DIR / "lstm_probabilities_timeline_train.png"

LOG_FILE = OUTPUT_DIR / "run.log"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Hyperparameters & threshold
best_params = {
    "window_size": 256,
    "stride": 32,
    "hidden_size": 129,
    "num_layers": 2,
    "dropout": 0.29282452624664446,
    "lr": 0.002456920963831831,
    "batch_size": 128,
    "threshold": 0.8651818138230819
}

# Feature pruning
def prune_correlated_features(df, feats, thresh=0.95):
    corr = df[feats].corr().abs()
    mask = np.triu(np.ones_like(corr), k=1).astype(bool)
    high_corr = corr.where(mask).stack().loc[lambda x: x > thresh].index
    to_drop = {f2 for f1, f2 in high_corr}
    return [f for f in feats if f not in to_drop]

# Dataset 
class WindowDataset(Dataset):
    def __init__(self, X, y, window_size, starts):
        self.X = X
        self.y = y
        self.ws = window_size
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        return self.X[s:s+self.ws], self.y[s:s+self.ws]

# Model
class LSTMClassifierLogits(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers>1 else 0
        )
        self.out = nn.Linear(hidden_size, 1)
        # initialize weights/biases
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.out(out).squeeze(-1)  

# Sequence evaluation
def evaluate_seq_predictions(y_true, pred_windows, starts, window_size, threshold):
    total = len(y_true)
    votes = np.zeros(total)
    counts = np.zeros(total)
    for pred, start in zip(pred_windows, starts):
        votes[start:start+window_size] += pred
        counts[start:start+window_size] += 1
    averaged = np.divide(votes, counts, out=np.zeros_like(votes), where=counts>0)
    binary = (averaged > threshold).astype(int)
    # metrics
    TP=FP=FN=N=0
    for yt, yp in zip(y_true, binary):
        if yt==0:
            N+=1
        if yp==1:
            if yt==1: TP+=1
            else: FP+=1
        elif yt==1:
            FN+=1
    precision_corr = (TP/(TP+FP)*(1-FP/N)) if (TP+FP>0 and N>0) else 0.0
    recall = TP/(TP+FN) if (TP+FN)>0 else 0.0
    b2 = 0.5**2
    f_beta = ((1+b2)*precision_corr*recall/(b2*precision_corr+recall)) if (precision_corr+recall)>0 else 0.0
    auc = roc_auc_score(y_true, averaged) if len(np.unique(y_true))>1 else 0.0
    cm = confusion_matrix(y_true, binary)
    return averaged, binary, {
        'f0.5': f_beta,
        'precision_corr': precision_corr,
        'recall': recall,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'n_samples': total,
        'prediction_stats': {
            'mean_pred': float(np.mean(averaged)),
            'std_pred': float(np.std(averaged)),
            'min_pred': float(np.min(averaged)),
            'max_pred': float(np.max(averaged)),
            'positive_rate': float(np.mean(binary))
        }
    }

# Load data
logger.info("Loading data...")
train_df = pd.read_parquet(TRAIN_PATH)
eval_df  = pd.read_parquet(EVAL_PATH)
features = pd.read_csv(TARGETS_PATH)["target_channels"].tolist()
input_feats = prune_correlated_features(train_df, features)
logger.info(f"{len(input_feats)} features after pruning.")

# Scale features
logger.info("Scaling features...")
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df[input_feats])
eval_scaled  = scaler.transform(eval_df[input_feats])

# Prepare data windows
X_tr = torch.from_numpy(train_scaled.astype(np.float32))
y_tr = torch.from_numpy(train_df["is_anomaly"].values.astype(np.float32))
N_tr = X_tr.shape[0]
ws = best_params["window_size"]
stride = min(best_params["stride"], max(1, N_tr-ws))
starts_tr = torch.cat([
    torch.arange(0, N_tr-ws, step=stride),
    torch.tensor([N_tr-ws], dtype=torch.long)
])
starts_tr = torch.unique(starts_tr, sorted=True)
# Use shuffle=True for training, and a separate non-shuffled loader for prediction
train_dl = DataLoader(WindowDataset(X_tr, y_tr, ws, starts_tr), batch_size=best_params["batch_size"], shuffle=True)

# Train model
pos_weight = torch.tensor([1.0/train_df.is_anomaly.mean()-1.0]).to(DEVICE)
model = LSTMClassifierLogits(len(input_feats), best_params["hidden_size"], best_params["num_layers"], best_params["dropout"]).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

logger.info("Starting training...")
model.train()
for epoch in range(1,21):
    total_loss=0.0; nb=0
    for xb,yb in train_dl:
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        optim.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        total_loss+=loss.item(); nb+=1
    avg_loss = total_loss/nb
    if epoch%5==0:
        model.eval()
        with torch.no_grad():
            sx, sy = next(iter(train_dl))
            logits = model(sx[:1].to(DEVICE))
            probs = torch.sigmoid(logits)
            logger.info(f"Epoch {epoch}/20 | Loss: {avg_loss:.4f} | Sample pred range: [{probs.min():.4f},{probs.max():.4f}] | Sample target rate: {sy[:1].mean():.4f}")
        model.train()
    else:
        logger.info(f"Epoch {epoch}/20 | Loss: {avg_loss:.4f}")

# Evaluation
X_ev = torch.from_numpy(eval_scaled.astype(np.float32))
y_ev = torch.from_numpy(eval_df["is_anomaly"].values.astype(np.float32))
M_ev = X_ev.shape[0]
starts_ev = torch.cat([
    torch.arange(0, M_ev-ws, step=stride),
    torch.tensor([M_ev-ws], dtype=torch.long)
])
starts_ev = torch.unique(starts_ev, sorted=True)
eval_dl = DataLoader(WindowDataset(X_ev, y_ev, ws, starts_ev), batch_size=best_params["batch_size"], shuffle=False)

model.eval()
pred_windows = []
with torch.no_grad():
    for xb, _ in eval_dl:
        logits = model(xb.to(DEVICE))
        preds = torch.sigmoid(logits).cpu().numpy()
        pred_windows.append(preds)
pred_windows = np.vstack(pred_windows)

averaged, binary, metrics = evaluate_seq_predictions(
    y_ev.numpy(), pred_windows, starts_ev.numpy(), ws, best_params["threshold"]
)

# Save outputs
torch.save(model.state_dict(), MODEL_STATE)
with open(RESULTS_PATH, 'w') as f:
    json.dump(metrics, f, indent=2)

# Event detection
eval_df = eval_df.copy()
eval_df['pred_anomaly'] = binary
eval_df['event_true'] = (eval_df['is_anomaly']==1)
mask = eval_df['event_true']
event_starts = mask & (~mask.shift(fill_value=False))
eval_df['event_id'] = np.where(mask, event_starts.cumsum(), np.nan)

events = []
for eid, grp in eval_df[mask].groupby('event_id'):
    length = int(len(grp))
    detected = bool(grp['pred_anomaly'].any())
    events.append({'event_id': int(eid), 'length': length, 'detected': detected})

total = len(events)
detected_count = sum(e['detected'] for e in events)
logger.info(f"[EVAL] Detected {detected_count} out of {total} events ({(detected_count/total*100 if total>0 else 0):.1f}%).")

# Save events JSON
with open(EVENTS_PATH, 'w') as f:
    json.dump({'split': 'eval', 'model': OUTPUT_DIR.name, 'threshold': best_params['threshold'], 'events': events}, f, indent=2)
logger.info(f"Saved {total} eval event records to {EVENTS_PATH}")

# Visualizations
if total > 0:
    df_ev = pd.DataFrame(events)[['length','detected']]
    max_len = int(df_ev['length'].max())
    bins = [1,1000,10000,20000,30000,60000, max_len+1]
    labels = ["1–1k","1k–10k","10k–20k","20k–30k","30k–60k",f"60k–{max_len}"]

    df_ev['length_bin'] = pd.cut(
        df_ev['length'], bins=bins, labels=labels,
        right=False, include_lowest=True
    )
    grouped = df_ev.groupby('length_bin')['detected'].agg(total_events='count', detected_events='sum')
    grouped['detection_rate'] = grouped['detected_events']/grouped['total_events']*100

    import matplotlib.pyplot as plt

    # Detection rate
    plt.figure(figsize=(8,4))
    plt.bar(grouped.index.astype(str), grouped['detection_rate'])
    plt.xlabel('Event Length Range')
    plt.ylabel('Detection Rate (%)')
    plt.title('LSTM Detection Rate by Event Length (Eval)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PLOT_LEN_RATE_EVAL, dpi=150)
    plt.close()

    # Detected vs missed counts
    undetected = grouped['total_events'] - grouped['detected_events']
    x = np.arange(len(grouped.index))
    width = 0.4
    plt.figure(figsize=(8,4))
    plt.bar(x-width/2, grouped['detected_events'], width, label='Detected')
    plt.bar(x+width/2, undetected, width, label='Missed')
    plt.xticks(x, grouped.index.astype(str), rotation=45, ha='right')
    plt.xlabel('Event Length Range')
    plt.ylabel('Number of Events')
    plt.title('LSTM Detected vs. Missed by Event Length (Eval)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_LEN_COUNTS_EVAL, dpi=150)
    plt.close()
else:
    logger.info("[EVAL] No true events; skipping length-based plots.")

# Probability timeline
idx_ev = np.arange(len(averaged))
plt.figure(figsize=(10,4))
plt.plot(idx_ev, averaged, label="P(anomaly)")
plt.scatter(idx_ev[eval_df['is_anomaly'].values==1], averaged[eval_df['is_anomaly'].values==1], marker='x', label='True anomaly')
fp_mask = (averaged>best_params['threshold']) & (eval_df['is_anomaly'].values==0)
plt.scatter(idx_ev[fp_mask], averaged[fp_mask], facecolors='none', edgecolors='r', label='False positive')
plt.axhline(best_params['threshold'], linestyle='--', label='Threshold')
plt.title("LSTM — Anomaly Probabilities (Eval)")
plt.xlabel("Sample Index")
plt.ylabel("Probability")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(PLOT_PROBS_EVAL, dpi=150)
plt.close()

#TRAIN predictions, events & plots
logger.info("Scoring TRAIN set for event detection and plots...")
train_pred_dl = DataLoader(WindowDataset(X_tr, y_tr, ws, starts_tr), batch_size=best_params["batch_size"], shuffle=False)
pred_windows_tr = []
with torch.no_grad():
    for xb, _ in train_pred_dl:
        logits = model(xb.to(DEVICE))
        preds = torch.sigmoid(logits).cpu().numpy()
        pred_windows_tr.append(preds)
pred_windows_tr = np.vstack(pred_windows_tr)

avg_tr, bin_tr, metrics_tr = evaluate_seq_predictions(
    y_tr.numpy(), pred_windows_tr, starts_tr.numpy(), ws, best_params["threshold"]
)

# Build events on TRAIN
train_df = train_df.copy()
train_df['pred_anomaly'] = bin_tr
train_df['event_true'] = (train_df['is_anomaly']==1)
mask_tr = train_df['event_true']
event_starts_tr = mask_tr & (~mask_tr.shift(fill_value=False))
train_df['event_id'] = np.where(mask_tr, event_starts_tr.cumsum(), np.nan)

events_tr = []
for eid, grp in train_df[mask_tr].groupby('event_id'):
    length = int(len(grp))
    detected = bool(grp['pred_anomaly'].any())
    events_tr.append({'event_id': int(eid), 'length': length, 'detected': detected})

tot_tr = len(events_tr)
det_tr = sum(e['detected'] for e in events_tr)
logger.info(f"[TRAIN] Detected {det_tr} out of {tot_tr} events ({(det_tr/tot_tr*100 if tot_tr>0 else 0):.1f}%).")

# Save events JSON (TRAIN)
with open(EVENTS_TRAIN_PATH, 'w') as f:
    json.dump({'split': 'train', 'model': OUTPUT_DIR.name, 'threshold': best_params['threshold'], 'events': events_tr}, f, indent=2)
logger.info(f"Saved {tot_tr} train event records to {EVENTS_TRAIN_PATH}")

# Plots (TRAIN) -> saved to disk
if tot_tr > 0:
    df_tr = pd.DataFrame(events_tr)[['length','detected']]
    max_len_tr = int(df_tr['length'].max())
    bins_tr = [1,1000,10000,20000,30000,60000, max_len_tr+1]
    labels_tr = ["1–1k","1k–10k","10k–20k","20k–30k","30k–60k",f"60k–{max_len_tr}"]

    df_tr['length_bin'] = pd.cut(
        df_tr['length'], bins=bins_tr, labels=labels_tr,
        right=False, include_lowest=True
    )
    grouped_tr = df_tr.groupby('length_bin')['detected'].agg(total_events='count', detected_events='sum')
    grouped_tr['detection_rate'] = grouped_tr['detected_events']/grouped_tr['total_events']*100

    # Detection rate (TRAIN)
    plt.figure(figsize=(8,4))
    plt.bar(grouped_tr.index.astype(str), grouped_tr['detection_rate'])
    plt.xlabel('Event Length Range')
    plt.ylabel('Detection Rate (%)')
    plt.title('LSTM Detection Rate by Event Length (Train)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PLOT_LEN_RATE_TRAIN, dpi=150)
    plt.close()

    # Detected vs missed counts (TRAIN)
    undetected_tr = grouped_tr['total_events'] - grouped_tr['detected_events']
    xtr = np.arange(len(grouped_tr.index))
    width = 0.4
    plt.figure(figsize=(8,4))
    plt.bar(xtr-width/2, grouped_tr['detected_events'], width, label='Detected')
    plt.bar(xtr+width/2, undetected_tr, width, label='Missed')
    plt.xticks(xtr, grouped_tr.index.astype(str), rotation=45, ha='right')
    plt.xlabel('Event Length Range')
    plt.ylabel('Number of Events')
    plt.title('LSTM Detected vs. Missed by Event Length (Train)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_LEN_COUNTS_TRAIN, dpi=150)
    plt.close()
else:
    logger.info("[TRAIN] No true events; skipping length-based plots.")

# Probability timeline (TRAIN)
idx_tr = np.arange(len(avg_tr))
plt.figure(figsize=(10,4))
plt.plot(idx_tr, avg_tr, label="P(anomaly)")
plt.scatter(idx_tr[train_df['is_anomaly'].values==1], avg_tr[train_df['is_anomaly'].values==1], marker='x', label='True anomaly')
fp_mask_tr = (avg_tr>best_params['threshold']) & (train_df['is_anomaly'].values==0)
plt.scatter(idx_tr[fp_mask_tr], avg_tr[fp_mask_tr], facecolors='none', edgecolors='r', label='False positive')
plt.axhline(best_params['threshold'], linestyle='--', label='Threshold')
plt.title("LSTM — Anomaly Probabilities (Train)")
plt.xlabel("Sample Index")
plt.ylabel("Probability")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(PLOT_PROBS_TRAIN, dpi=150)
plt.close()

logger.info("All artifacts saved to disk.")
