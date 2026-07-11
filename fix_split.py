import numpy as np
import os

DATA_DIR   = r"C:\ids_project\data"
NUM_CLIENTS = 5

print("Fixing data split to IID...")

X_train = []
y_train = []
for i in range(NUM_CLIENTS):
    X_train.append(
        np.load(f"{DATA_DIR}/client_{i}/X_train.npy")
    )
    y_train.append(
        np.load(f"{DATA_DIR}/client_{i}/y_train.npy")
    )

X_all = np.vstack(X_train)
y_all = np.concatenate(y_train)

# Shuffle
idx = np.random.permutation(len(X_all))
X_all = X_all[idx]
y_all = y_all[idx]

# Equal IID split across 5 clients
splits_X = np.array_split(X_all, NUM_CLIENTS)
splits_y = np.array_split(y_all, NUM_CLIENTS)

for i in range(NUM_CLIENTS):
    np.save(f"{DATA_DIR}/client_{i}/X_train.npy", splits_X[i])
    np.save(f"{DATA_DIR}/client_{i}/y_train.npy", splits_y[i])
    classes_in_client = len(np.unique(splits_y[i]))
    print(f"Client {i}: {len(splits_X[i]):,} samples | "
          f"{classes_in_client} classes")

print("\n✅ IID split done!")
print("Now run: python models\\federated_learning.py")