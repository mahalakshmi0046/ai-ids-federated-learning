# ============================================================
# BASELINE COMPARISON — FL vs Traditional ML
# Proves federated learning works with minimal accuracy cost
# This is the KEY research contribution proof!
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import seaborn as sns
warnings.filterwarnings('ignore')

print("=" * 60)
print("  Baseline Comparison — FL vs Traditional ML")
print("=" * 60)

DATA_DIR   = r"C:\ids_project\data"
MODEL_DIR  = r"C:\ids_project\models"
RESULT_DIR = r"C:\ids_project\results\comparison"
os.makedirs(RESULT_DIR, exist_ok=True)

# Load data
print("\n📂 Loading data...")
X_test = np.load(f"{DATA_DIR}/X_test.npy")
y_test = np.load(f"{DATA_DIR}/y_test.npy")

with open(f"{DATA_DIR}/classes.txt",
          encoding="utf-8") as f:
    classes = f.read().strip().split("\n")

benign_idx  = classes.index("BENIGN")
NUM_CLASSES = len(classes)

# Load all training data
X_all, y_all = [], []
for i in range(5):
    X_all.append(
        np.load(f"{DATA_DIR}/client_{i}/X_train.npy")
    )
    y_all.append(
        np.load(f"{DATA_DIR}/client_{i}/y_train.npy")
    )

X_train = np.vstack(X_all)
y_train = np.concatenate(y_all)

print(f"   Train: {len(X_train):,}")
print(f"   Test:  {len(X_test):,}")
print(f"   Classes: {NUM_CLASSES}")

# ── Balanced test set ────────────────────────────────────────
from collections import Counter
test_idx = []
for cls in range(NUM_CLASSES):
    idx = np.where(y_test == cls)[0]
    if len(idx) > 0:
        n = min(len(idx), 500)
        test_idx.extend(
            np.random.choice(idx, n, replace=False)
        )
test_idx   = np.array(test_idx)
X_test_bal = X_test[test_idx]
y_test_bal = y_test[test_idx]

print(f"\n✅ Balanced test: {len(X_test_bal):,} samples")

results = {}

# ── Model 1: No FL — Single Client Only ─────────────────────
print("\n" + "="*50)
print("Model 1: No FL — Single Client (Local Only)")
print("="*50)
print("Simulates: One organization training alone")

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

def balance_data(X, y, target=2000):
    X_b, y_b = [], []
    for cls in range(NUM_CLASSES):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        n       = min(target, max(1, len(idx)))
        sampled = idx[np.random.choice(
            len(idx), n,
            replace=len(idx) < n
        )]
        X_b.append(X[sampled])
        y_b.append(np.full(n, cls))
    X_b = np.vstack(X_b)
    y_b = np.concatenate(y_b)
    idx = np.random.permutation(len(X_b))
    return X_b[idx], y_b[idx]

def train_model(X, y, epochs=10,
                device=torch.device("cpu")):
    X_b, y_b  = balance_data(X, y)
    model      = IDSNet(X.shape[1], NUM_CLASSES).to(device)
    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.Adam(
        model.parameters(), lr=0.001
    )
    dataset    = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_b),
        torch.LongTensor(y_b)
    )
    loader     = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=True
    )
    model.train()
    for _ in range(epochs):
        for bX, by in loader:
            bX, by = bX.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bX), by)
            loss.backward()
            optimizer.step()
    return model

def evaluate_model(model, X, y,
                   device=torch.device("cpu")):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            b = torch.FloatTensor(
                X[i:i+512]
            ).to(device)
            preds.extend(
                model(b).argmax(dim=1).cpu().numpy()
            )
    preds      = np.array(preds)
    acc        = accuracy_score(y, preds)
    f1         = f1_score(y, preds,
                          average='weighted',
                          zero_division=0)
    y_bin      = (y != benign_idx).astype(int)
    p_bin      = (preds != benign_idx).astype(int)
    bin_acc    = accuracy_score(y_bin, p_bin)
    bin_f1     = f1_score(y_bin, p_bin,
                          zero_division=0)
    bin_prec   = precision_score(y_bin, p_bin,
                                 zero_division=0)
    bin_rec    = recall_score(y_bin, p_bin,
                              zero_division=0)
    return {
        "multi_acc": acc,
        "multi_f1":  f1,
        "bin_acc":   bin_acc,
        "bin_f1":    bin_f1,
        "bin_prec":  bin_prec,
        "bin_rec":   bin_rec,
        "preds":     preds
    }

# Train on client 0 only
X_c0 = np.load(f"{DATA_DIR}/client_0/X_train.npy")
y_c0 = np.load(f"{DATA_DIR}/client_0/y_train.npy")

print("Training on single client data...")
local_model  = train_model(X_c0, y_c0, epochs=10)
local_result = evaluate_model(
    local_model, X_test_bal, y_test_bal
)
results["No FL\n(Local Only)"] = local_result

print(f"  Multi-class Accuracy : "
      f"{local_result['multi_acc']*100:.2f}%")
print(f"  Binary F1-Score      : "
      f"{local_result['bin_f1']*100:.2f}%")

# ── Model 2: Centralized (No Privacy) ───────────────────────
print("\n" + "="*50)
print("Model 2: Centralized ML (No Privacy)")
print("="*50)
print("Simulates: All orgs share raw data (privacy violated)")
print("This is what traditional IDS does")

print("Training Random Forest on ALL data...")
# Sample for speed
sample_idx = np.random.choice(
    len(X_train), min(50000, len(X_train)), replace=False
)
X_s = X_train[sample_idx]
y_s = y_train[sample_idx]

# Balance
X_sb, y_sb = balance_data(X_s, y_s, target=500)

rf = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1
)
rf.fit(X_sb, y_sb)
rf_preds = rf.predict(X_test_bal)

central_result = {
    "multi_acc": accuracy_score(y_test_bal, rf_preds),
    "multi_f1":  f1_score(y_test_bal, rf_preds,
                           average='weighted',
                           zero_division=0),
    "bin_acc":   accuracy_score(
                     (y_test_bal != benign_idx).astype(int),
                     (rf_preds != benign_idx).astype(int)
                 ),
    "bin_f1":    f1_score(
                     (y_test_bal != benign_idx).astype(int),
                     (rf_preds != benign_idx).astype(int),
                     zero_division=0
                 ),
    "bin_prec":  precision_score(
                     (y_test_bal != benign_idx).astype(int),
                     (rf_preds != benign_idx).astype(int),
                     zero_division=0
                 ),
    "bin_rec":   recall_score(
                     (y_test_bal != benign_idx).astype(int),
                     (rf_preds != benign_idx).astype(int),
                     zero_division=0
                 ),
    "preds":     rf_preds
}
results["Centralized RF\n(No Privacy)"] = central_result

print(f"  Multi-class Accuracy : "
      f"{central_result['multi_acc']*100:.2f}%")
print(f"  Binary F1-Score      : "
      f"{central_result['bin_f1']*100:.2f}%")

# ── Model 3: Your FL Model ───────────────────────────────────
print("\n" + "="*50)
print("Model 3: YOUR FL Model (Privacy Preserved)")
print("="*50)
print("5 clients, FedAvg, DP ε=0.000484")

fl_model = IDSNet(
    X_train.shape[1], NUM_CLASSES
).to(torch.device("cpu"))

fl_path = f"{MODEL_DIR}/fl_global_model.pth"
fl_model.load_state_dict(
    torch.load(fl_path, map_location="cpu")
)

fl_result = evaluate_model(
    fl_model, X_test_bal, y_test_bal
)
results["Your FL Model\n(DP ε=0.000484)"] = fl_result

print(f"  Multi-class Accuracy : "
      f"{fl_result['multi_acc']*100:.2f}%")
print(f"  Binary F1-Score      : "
      f"{fl_result['bin_f1']*100:.2f}%")

# ── Comparison Table ─────────────────────────────────────────
print("\n" + "="*60)
print("📊 COMPARISON RESULTS")
print("="*60)

models   = list(results.keys())
metrics  = {
    "Multi-class\nAccuracy": [
        f"{r['multi_acc']*100:.2f}%"
        for r in results.values()
    ],
    "Binary\nAccuracy": [
        f"{r['bin_acc']*100:.2f}%"
        for r in results.values()
    ],
    "Binary\nF1-Score": [
        f"{r['bin_f1']*100:.2f}%"
        for r in results.values()
    ],
    "Precision": [
        f"{r['bin_prec']*100:.2f}%"
        for r in results.values()
    ],
    "Recall": [
        f"{r['bin_rec']*100:.2f}%"
        for r in results.values()
    ],
    "Privacy": [
        "❌ None",
        "❌ None",
        "✅ DP ε=0.000484"
    ],
    "Data Sharing": [
        "Local only",
        "❌ All data shared",
        "✅ Weights only"
    ]
}

df = pd.DataFrame(metrics, index=models)
print(df.to_string())

# Save to CSV
df.to_csv(f"{RESULT_DIR}/comparison_table.csv")
print(f"\n✅ Saved: comparison_table.csv")

# ── Plot 1: Accuracy Comparison Bar Chart ───────────────────
print("\n📈 Generating comparison charts...")

model_names = [
    "No FL\n(Local Only)",
    "Centralized RF\n(No Privacy)",
    "Your FL Model\n(DP ε=0.000484)"
]
colors = ["#64748b", "#ef4444", "#10b981"]

multi_accs = [
    r["multi_acc"] * 100 for r in results.values()
]
bin_f1s    = [
    r["bin_f1"] * 100 for r in results.values()
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#0a0e1a")

for ax in axes:
    ax.set_facecolor("#0d1f35")
    ax.tick_params(colors="#e2e8f0")
    ax.spines[:].set_color("#1e4a8a")

# Multi-class accuracy
bars1 = axes[0].bar(
    model_names, multi_accs,
    color=colors, width=0.5, edgecolor="#1e4a8a"
)
axes[0].set_title(
    "Multi-class Accuracy Comparison",
    color="#e2e8f0", fontsize=13, fontweight="bold"
)
axes[0].set_ylabel("Accuracy (%)", color="#e2e8f0")
axes[0].set_ylim(0, 110)
axes[0].yaxis.label.set_color("#e2e8f0")

for bar, val in zip(bars1, multi_accs):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{val:.1f}%",
        ha="center", va="bottom",
        color="#e2e8f0", fontweight="bold"
    )

# Binary F1
bars2 = axes[1].bar(
    model_names, bin_f1s,
    color=colors, width=0.5, edgecolor="#1e4a8a"
)
axes[1].set_title(
    "Binary F1-Score Comparison",
    color="#e2e8f0", fontsize=13, fontweight="bold"
)
axes[1].set_ylabel("F1-Score (%)", color="#e2e8f0")
axes[1].set_ylim(0, 110)
axes[1].yaxis.label.set_color("#e2e8f0")

for bar, val in zip(bars2, bin_f1s):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{val:.1f}%",
        ha="center", va="bottom",
        color="#e2e8f0", fontweight="bold"
    )

# Add privacy annotation
axes[1].annotate(
    "✅ Privacy\nPreserved",
    xy=(2, bin_f1s[2]),
    xytext=(1.5, bin_f1s[2] + 8),
    color="#10b981", fontsize=10,
    arrowprops=dict(color="#10b981", arrowstyle="->")
)

plt.tight_layout()
plt.savefig(
    f"{RESULT_DIR}/accuracy_comparison.png",
    dpi=150, facecolor="#0a0e1a"
)
plt.close()
print("   ✅ Saved: accuracy_comparison.png")

# ── Plot 2: Privacy-Accuracy Tradeoff ───────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("#0a0e1a")
ax.set_facecolor("#0d1f35")
ax.tick_params(colors="#e2e8f0")
ax.spines[:].set_color("#1e4a8a")

privacy_levels = [0, 0, 0.000484]
accuracies     = [
    r["multi_acc"] * 100 for r in results.values()
]

scatter_colors = ["#64748b", "#ef4444", "#10b981"]
labels         = [
    "No FL (Local)", "Centralized RF", "Your FL + DP"
]

for x, y, c, l in zip(
    privacy_levels, accuracies,
    scatter_colors, labels
):
    ax.scatter(
        x, y, color=c, s=300,
        zorder=5, label=l
    )
    ax.annotate(
        f"{l}\n{y:.1f}%",
        (x, y),
        textcoords="offset points",
        xytext=(10, 5),
        color=c, fontsize=10
    )

ax.set_xlabel(
    "Privacy Guarantee (DP ε — lower = more private)",
    color="#e2e8f0"
)
ax.set_ylabel("Accuracy (%)", color="#e2e8f0")
ax.set_title(
    "Privacy vs Accuracy Tradeoff\n"
    "Our FL system achieves high accuracy WITH privacy",
    color="#e2e8f0", fontsize=13, fontweight="bold"
)
ax.legend(
    facecolor="#0d1f35",
    labelcolor="#e2e8f0",
    edgecolor="#1e4a8a"
)
ax.xaxis.label.set_color("#e2e8f0")

plt.tight_layout()
plt.savefig(
    f"{RESULT_DIR}/privacy_accuracy_tradeoff.png",
    dpi=150, facecolor="#0a0e1a"
)
plt.close()
print("   ✅ Saved: privacy_accuracy_tradeoff.png")

# ── Plot 3: Confusion Matrix for FL Model ───────────────────
cm = confusion_matrix(y_test_bal, fl_result["preds"])
plt.figure(figsize=(12, 10))
plt.gca().set_facecolor("#0d1f35")
plt.gcf().patch.set_facecolor("#0a0e1a")

short_names = [
    c.replace("Web Attack  ", "WA-")
     .replace("DoS ", "DoS-")
    for c in classes
]

sns.heatmap(
    cm, annot=True, fmt="d",
    cmap="Blues",
    xticklabels=short_names,
    yticklabels=short_names,
    linewidths=0.5
)
plt.title(
    "FL Model — Confusion Matrix",
    color="#e2e8f0", fontsize=14,
    fontweight="bold", pad=15
)
plt.ylabel("Actual", color="#e2e8f0")
plt.xlabel("Predicted", color="#e2e8f0")
plt.xticks(
    rotation=45, ha="right",
    color="#e2e8f0", fontsize=8
)
plt.yticks(color="#e2e8f0", fontsize=8)
plt.tight_layout()
plt.savefig(
    f"{RESULT_DIR}/confusion_matrix_fl.png",
    dpi=150, facecolor="#0a0e1a"
)
plt.close()
print("   ✅ Saved: confusion_matrix_fl.png")

# ── Final Summary ────────────────────────────────────────────
print("\n" + "="*60)
print("✅ COMPARISON COMPLETE!")
print("="*60)

local_acc   = list(results.values())[0]["multi_acc"]
central_acc = list(results.values())[1]["multi_acc"]
fl_acc      = list(results.values())[2]["multi_acc"]
fl_f1       = list(results.values())[2]["bin_f1"]

gap = central_acc - fl_acc

print(f"\n  No FL (Local Only)      : "
      f"{local_acc*100:.2f}%")
print(f"  Centralized RF          : "
      f"{central_acc*100:.2f}%")
print(f"  Your FL + DP            : "
      f"{fl_acc*100:.2f}%")
print(f"\n  Privacy-Accuracy Gap    : "
      f"{gap*100:.2f}%")
print(f"  Binary F1-Score         : "
      f"{fl_f1*100:.2f}%")
print(f"\n  ✅ FL achieves {fl_acc*100:.1f}% accuracy")
print(f"     WITH privacy guarantee (ε=0.000484)")
print(f"     vs {central_acc*100:.1f}% centralized "
      f"(no privacy)")
print(f"     Cost of privacy: only {gap*100:.2f}%!")
print(f"\n  Saved to: {RESULT_DIR}/")
print("="*60)
