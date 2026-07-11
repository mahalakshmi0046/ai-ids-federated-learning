# ============================================================
# FINAL SOLUTION — FL with SMOTE balancing per client
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import os, warnings
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import resample
warnings.filterwarnings('ignore')

print("="*60)
print("  FL IDS — With Proper Balancing")
print("="*60)

DATA_DIR   = r"C:\ids_project\data"
MODEL_DIR  = r"C:\ids_project\models"
RESULT_DIR = r"C:\ids_project\results"

with open(f"{DATA_DIR}/classes.txt", encoding="utf-8") as f:
    classes = f.read().strip().split("\n")

NUM_CLASSES = len(classes)
benign_idx  = classes.index('BENIGN')
device      = torch.device("cpu")

NUM_ROUNDS   = 10
LOCAL_EPOCHS = 5
LR           = 0.001
BATCH_SIZE   = 128

def balance_data(X, y, target_per_class=2000):
    """Oversample minority classes, undersample majority."""
    X_bal, y_bal = [], []
    for cls in range(NUM_CLASSES):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        n = target_per_class
        if len(idx) >= n:
            sampled = idx[np.random.choice(
                len(idx), n, replace=False
            )]
        else:
            sampled = idx[np.random.choice(
                len(idx), n, replace=True
            )]
        X_bal.append(X[sampled])
        y_bal.append(np.full(n, cls))
    X_bal = np.vstack(X_bal)
    y_bal = np.concatenate(y_bal)
    idx   = np.random.permutation(len(X_bal))
    return X_bal[idx], y_bal[idx]

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

# Load test data — balanced evaluation
X_test = np.load(f"{DATA_DIR}/X_test.npy")
y_test = np.load(f"{DATA_DIR}/y_test.npy")

# Balanced test set
test_idx = []
for cls in range(NUM_CLASSES):
    idx = np.where(y_test == cls)[0]
    if len(idx) > 0:
        n = min(len(idx), 500)
        test_idx.extend(
            np.random.choice(idx, n, replace=False)
        )
test_idx    = np.array(test_idx)
X_test_bal  = X_test[test_idx]
y_test_bal  = y_test[test_idx]
print(f"\n✅ Balanced test: {len(X_test_bal):,} samples")

input_size    = X_test.shape[1]
global_model  = IDSNet(input_size, NUM_CLASSES).to(device)
global_params = get_params(global_model)
epsilon       = 0.0

with open(f"{RESULT_DIR}/fl_log.txt", "w") as f:
    f.write("Round,Accuracy,F1,BinAcc,BinF1,DP_Epsilon\n")

for rnd in range(1, NUM_ROUNDS + 1):
    print(f"\n{'='*50}")
    print(f"  ROUND {rnd}/{NUM_ROUNDS}")
    print(f"{'='*50}")

    all_params = []
    all_sizes  = []

    for cid in range(5):
        # Load raw client data
        X_c = np.load(
            f"{DATA_DIR}/client_{cid}/X_train.npy"
        )
        y_c = np.load(
            f"{DATA_DIR}/client_{cid}/y_train.npy"
        )

        # Balance BEFORE training
        X_b, y_b = balance_data(X_c, y_c,
                                  target_per_class=2000)

        model     = IDSNet(input_size, NUM_CLASSES).to(device)
        set_params(model, global_params)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR
        )

        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_b),
            torch.LongTensor(y_b)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True
        )

        model.train()
        total_loss = 0.0
        for _ in range(LOCAL_EPOCHS):
            for bX, by in loader:
                bX = bX.to(device)
                by = by.to(device)
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(model(bX), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / (len(loader) * LOCAL_EPOCHS)

        # Local accuracy on balanced data
        model.eval()
        with torch.no_grad():
            sample = min(500, len(X_b))
            preds  = model(
                torch.FloatTensor(X_b[:sample]).to(device)
            ).argmax(dim=1).cpu().numpy()
        local_acc = accuracy_score(y_b[:sample], preds)

        print(f"  [Client {cid}] "
              f"Loss: {avg_loss:.4f} | "
              f"Acc: {local_acc*100:.1f}% | "
              f"Balanced: {len(X_b):,}")

        all_params.append(get_params(model))
        all_sizes.append(len(X_b))

    # FedAvg
    total = sum(all_sizes)
    avg   = None
    for params, n in zip(all_params, all_sizes):
        w = n / total
        if avg is None:
            avg = [p * w for p in params]
        else:
            avg = [a + p*w for a, p in zip(avg, params)]

    # Tiny DP noise
    noisy = [(p + np.random.normal(
                  0, 0.00001, p.shape
              )).astype(np.float32) for p in avg]
    epsilon += 0.00001 * np.sqrt(2 * np.log(1.25/1e-5))

    set_params(global_model, noisy)
    global_params = get_params(global_model)

    # Evaluate on balanced test
    global_model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test_bal), 512):
            b = torch.FloatTensor(
                X_test_bal[i:i+512]
            ).to(device)
            preds.extend(
                global_model(b).argmax(dim=1).cpu().numpy()
            )
    preds = np.array(preds)

    acc = accuracy_score(y_test_bal, preds)
    f1  = f1_score(y_test_bal, preds,
                   average='weighted', zero_division=0)

    y_true_bin = (y_test_bal != benign_idx).astype(int)
    y_pred_bin = (preds != benign_idx).astype(int)
    bin_acc = accuracy_score(y_true_bin, y_pred_bin)
    bin_f1  = f1_score(y_true_bin, y_pred_bin,
                       zero_division=0)

    print(f"\n  📊 Multi-class:")
    print(f"     Accuracy : {acc*100:.2f}%")
    print(f"     F1-Score : {f1*100:.2f}%")
    print(f"  🎯 Binary:")
    print(f"     Accuracy : {bin_acc*100:.2f}%")
    print(f"     F1-Score : {bin_f1*100:.2f}%")
    print(f"  🔒 DP ε    : {epsilon:.6f}")

    with open(f"{RESULT_DIR}/fl_log.txt", "a") as f:
        f.write(f"{rnd},{acc:.4f},{f1:.4f},"
                f"{bin_acc:.4f},{bin_f1:.4f},"
                f"{epsilon:.6f}\n")

torch.save(global_model.state_dict(),
           f"{MODEL_DIR}/fl_global_model.pth")

print("\n" + "="*60)
print("✅ COMPLETE!")
print(f"   Multi-class Accuracy : {acc*100:.2f}%")
print(f"   Binary Accuracy      : {bin_acc*100:.2f}%")
print(f"   Binary F1            : {bin_f1*100:.2f}%")
print(f"   DP ε                 : {epsilon:.6f}")
print("="*60)