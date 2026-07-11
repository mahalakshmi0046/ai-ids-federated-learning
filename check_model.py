import numpy as np
import torch
import torch.nn as nn
import pickle, os

DATA_DIR  = r"C:\ids_project\data"
MODEL_DIR = r"C:\ids_project\models"

with open(f"{DATA_DIR}/classes.txt", encoding="utf-8") as f:
    classes = f.read().strip().split("\n")

NUM_CLASSES = len(classes)
print(f"Classes: {NUM_CLASSES}")
print(f"Class names: {classes}")

# Load test data
X_test = np.load(f"{DATA_DIR}/X_test.npy")
y_test = np.load(f"{DATA_DIR}/y_test.npy")

print(f"\nTest set size: {len(X_test):,}")
print(f"Class distribution in test set:")
unique, counts = np.unique(y_test, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {classes[u]:40s}: {c:,} ({c/len(y_test)*100:.1f}%)")