import numpy as np
import torch
import torch.nn as nn
import os

DATA_DIR  = r"C:\ids_project\data"
MODEL_DIR = r"C:\ids_project\models"

with open(f"{DATA_DIR}/classes.txt", encoding="utf-8") as f:
    classes = f.read().strip().split("\n")

NUM_CLASSES = len(classes)
benign_idx  = classes.index('BENIGN')

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

# Load one client's data and test locally
X = np.load(f"{DATA_DIR}/client_0/X_train.npy")
y = np.load(f"{DATA_DIR}/client_0/y_train.npy")

print(f"Client 0 data: {len(X):,} samples")
print(f"Classes in client 0:")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {classes[u]:35s}: {c:,}")

# Train a simple model on just client 0 data
model     = IDSNet(X.shape[1], NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Small quick training
idx     = np.random.choice(len(X), min(5000, len(X)), replace=False)
X_small = torch.FloatTensor(X[idx])
y_small = torch.LongTensor(y[idx])

print(f"\nTraining on 5000 samples for 10 epochs...")
model.train()
for epoch in range(10):
    optimizer.zero_grad()
    out  = model(X_small)
    loss = criterion(out, y_small)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 5 == 0:
        preds = out.argmax(dim=1).detach().numpy()
        acc   = (preds == y[idx]).mean()
        print(f"  Epoch {epoch+1}: Loss={loss.item():.4f} Acc={acc*100:.1f}%")

# Now test on test set
X_test = np.load(f"{DATA_DIR}/X_test.npy")
y_test = np.load(f"{DATA_DIR}/y_test.npy")

# Take small balanced sample
test_idx = []
for cls in range(NUM_CLASSES):
    cls_idx = np.where(y_test == cls)[0]
    if len(cls_idx) > 0:
        n = min(len(cls_idx), 200)
        test_idx.extend(
            np.random.choice(cls_idx, n, replace=False)
        )

test_idx = np.array(test_idx)
X_t = torch.FloatTensor(X_test[test_idx])
y_t = y_test[test_idx]

model.eval()
with torch.no_grad():
    preds = model(X_t).argmax(dim=1).numpy()

from sklearn.metrics import accuracy_score, f1_score
acc = accuracy_score(y_t, preds)
f1  = f1_score(y_t, preds, average='weighted', zero_division=0)

print(f"\n📊 Single client model on balanced test:")
print(f"   Accuracy : {acc*100:.2f}%")
print(f"   F1-Score : {f1*100:.2f}%")

print(f"\nPrediction distribution:")
unique, counts = np.unique(preds, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {classes[u]:35s}: {c}")