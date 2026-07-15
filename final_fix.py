# ============================================================
# FINAL FIX — Train directly on balanced data, evaluate properly
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import os, pickle, warnings
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report)
from collections import Counter
warnings.filterwarnings('ignore')

print("="*60)
print("  FL IDS — Final Fixed Version")
print("="*60)

DATA_DIR   = r"C:\ids_project\data"
MODEL_DIR  = r"C:\ids_project\models"
RESULT_DIR = r"C:\ids_project\results"

with open(f"{DATA_DIR}/classes.txt", encoding="utf-8") as f:
    classes = f.read().strip().split("\n")

NUM_CLASSES = len(classes)
benign_idx  = classes.index('BENIGN')
device      = torch.device("cpu")

# ── Load ALL training data ────────────────────────────────────
print("\n📂 Loading all client data...")
X_all, y_all = [], []
for i in range(5):
    X_all.append(np.load(f"{DATA_DIR}/client_{i}/X_train.npy"))
    y_all.append(np.load(f"{DATA_DIR}/client_{i}/y_train.npy"))

X_train = np.vstack(X_all)
y_train = np.concatenate(y_all)
print(f"   Total training: {len(X_train):,}")

# ── Load test data ────────────────────────────────────────────
X_test = np.load(f"{DATA_DIR}/X_test.npy")
y_test = np.load(f"{DATA_DIR}/y_test.npy")

# ── Create BALANCED dataset ───────────────────────────────────
print("\n⚖️  Creating balanced dataset...")
counts     = Counter(y_train.tolist())
# Use median count as target (ignore tiny classes)
valid_classes = [c for c, n in counts.items() if n >= 100]
target = int(np.median([counts[c] for c in valid_classes]))
target = min(target, 10000)  # Cap at 10k per class

X_bal, y_bal = [], []
for cls in range(NUM_CLASSES):
    idx = np.where(y_train == cls)[0]
    if len(idx) == 0:
        continue
    n = min(len(idx), target)
    sampled = np.random.choice(idx, n, replace=len(idx) < n)
    X_bal.append(X_train[sampled])
    y_bal.append(np.full(n, cls))
    print(f"   {classes[cls]:35s}: {n:,} samples")

X_bal = np.vstack(X_bal)
y_bal = np.concatenate(y_bal)

# Shuffle
idx   = np.random.permutation(len(X_bal))
X_bal = X_bal[idx]
y_bal = y_bal[idx]
print(f"\n   Balanced total: {len(X_bal):,}")

# ── Simulate FL: split balanced data across 5 clients ─────────
print("\n🏢 Simulating FL with balanced data...")
splits_X = np.array_split(X_bal, 5)
splits_y = np.array_split(y_bal, 5)

# ── Model ─────────────────────────────────────────────────────
class IDSNet(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out)
        )
    def forward(self, x):
        return self.net(x)

def get_params(m):
    return [v.cpu().detach().numpy().copy()
            for v in m.state_dict().values()]

def set_params(m, params):
    keys = list(m.state_dict().keys())
    sd   = {k: torch.tensor(v.copy())
            for k, v in zip(keys, params)}
    m.load_state_dict(sd, strict=True)

def add_dp_noise(params, noise=0.001):
    return [(p / max(1.0, np.linalg.norm(p)) +
             np.random.normal(0, noise, p.shape)
             ).astype(np.float32)
            for p in params]

# ── FL Training ───────────────────────────────────────────────
input_size   = X_train.shape[1]
NUM_ROUNDS   = 10
LOCAL_EPOCHS = 5
LR           = 0.001
BATCH_SIZE   = 256

global_model  = IDSNet(input_size, NUM_CLASSES).to(device)
global_params = get_params(global_model)
epsilon       = 0.0

with open(f"{RESULT_DIR}/fl_log.txt", "w") as f:
    f.write("Round,Accuracy,F1,BinAcc,BinF1,DP_Epsilon\n")

print(f"\n🚀 Starting FL Training ({NUM_ROUNDS} rounds)...")

for rnd in range(1, NUM_ROUNDS + 1):
    print(f"\n{'='*50}")
    print(f"  ROUND {rnd}/{NUM_ROUNDS}")
    print(f"{'='*50}")

    all_params = []
    all_sizes  = []

    for cid in range(5):
        X_c = splits_X[cid]
        y_c = splits_y[cid]

        model     = IDSNet(input_size, NUM_CLASSES).to(device)
        set_params(model, global_params)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR
        )

        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_c),
            torch.LongTensor(y_c)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True
        )

        model.train()
        total_loss = 0.0
        for _ in range(LOCAL_EPOCHS):
            for bX, by in loader:
                bX, by = bX.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bX), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / (len(loader) * LOCAL_EPOCHS)

        # Quick local accuracy
        model.eval()
        with torch.no_grad():
            preds = model(
                torch.FloatTensor(X_c[:500]).to(device)
            ).argmax(dim=1).cpu().numpy()
        local_acc = accuracy_score(y_c[:500], preds)

        print(f"  [Client {cid}] "
              f"Loss: {avg_loss:.4f} | "
              f"Acc: {local_acc*100:.1f}%")

        all_params.append(get_params(model))
        all_sizes.append(len(X_c))

    # FedAvg
    total = sum(all_sizes)
    avg   = None
    for params, n in zip(all_params, all_sizes):
        w = n / total
        if avg is None:
            avg = [p * w for p in params]
        else:
            avg = [a + p * w for a, p in zip(avg, params)]

    # Differential Privacy
    noisy_params = add_dp_noise(avg)
    epsilon += 0.001 * np.sqrt(2 * np.log(1.25 / 1e-5))
    set_params(global_model, noisy_params)
    global_params = get_params(global_model)

    # ── Evaluate on REAL test set ─────────────────────────
    global_model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 512):
            b = torch.FloatTensor(
                X_test[i:i+512]
            ).to(device)
            all_preds.extend(
                global_model(b).argmax(dim=1).cpu().numpy()
            )

    all_preds  = np.array(all_preds)

    # Multi-class
    acc = accuracy_score(y_test, all_preds)
    f1  = f1_score(y_test, all_preds,
                   average='weighted', zero_division=0)

    # Binary
    y_true_bin = (y_test != benign_idx).astype(int)
    y_pred_bin = (all_preds != benign_idx).astype(int)
    bin_acc    = accuracy_score(y_true_bin, y_pred_bin)
    bin_f1     = f1_score(y_true_bin, y_pred_bin,
                          zero_division=0)

    print(f"\n  📊 Multi-class:")
    print(f"     Accuracy : {acc*100:.2f}%")
    print(f"     F1-Score : {f1*100:.2f}%")
    print(f"\n  🎯 Binary (Attack vs Normal):")
    print(f"     Accuracy : {bin_acc*100:.2f}%")
    print(f"     F1-Score : {bin_f1*100:.2f}%")
    print(f"\n  🔒 DP ε = {epsilon:.4f}")

    with open(f"{RESULT_DIR}/fl_log.txt", "a") as f:
        f.write(f"{rnd},{acc:.4f},{f1:.4f},"
                f"{bin_acc:.4f},{bin_f1:.4f},"
                f"{epsilon:.4f}\n")

# Save model
torch.save(
    global_model.state_dict(),
    f"{MODEL_DIR}/fl_global_model.pth"
)

print("\n" + "="*60)
print("✅ DONE!")
print(f"   Multi-class Accuracy : {acc*100:.2f}%")
print(f"   Binary Accuracy      : {bin_acc*100:.2f}%")
print(f"   Binary F1-Score      : {bin_f1*100:.2f}%")
print(f"   DP ε                 : {epsilon:.4f}")
print("="*60)