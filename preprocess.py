# ============================================================
# PREPROCESSING — CICIDS2017 Dataset
# Loads all 8 CSV files, cleans, creates non-IID client splits
# ============================================================

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

print("="*60)
print("  AI-Powered IDS — Data Preprocessing")
print("="*60)

RAW_DIR    = r"C:\ids_project\raw_data"
OUTPUT_DIR = r"C:\ids_project\data"
NUM_CLIENTS = 5
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
for i in range(NUM_CLIENTS):
    os.makedirs(f"{OUTPUT_DIR}/client_{i}", exist_ok=True)

# ── Step 1: Load all CSV files ───────────────────────────────
print("\n📂 Loading all CSV files...")
all_dfs = []
for f in os.listdir(RAW_DIR):
    if f.endswith(".csv"):
        path = os.path.join(RAW_DIR, f)
        print(f"   Loading {f}...")
        df = pd.read_csv(path, low_memory=False)
        all_dfs.append(df)

df = pd.concat(all_dfs, ignore_index=True)
print(f"\n✅ Total rows loaded: {len(df):,}")

# ── Step 2: Clean ────────────────────────────────────────────
print("\n🧹 Cleaning data...")
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"   Rows after cleaning: {len(df):,}")

# ── Step 3: Labels ───────────────────────────────────────────
print("\n🏷️  Attack types found:")
print(df['Label'].value_counts())

# Simplify labels
df['Label'] = df['Label'].str.strip()
le = LabelEncoder()
y = le.fit_transform(df['Label'])
print(f"\n   Classes: {list(le.classes_)}")

# ── Step 4: Features ─────────────────────────────────────────
drop_cols = ['Label', 'Flow ID', 'Source IP', 'Destination IP',
             'Source Port', 'Timestamp']
feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in drop_cols]
X = df[feature_cols].values
print(f"\n✅ Features selected: {len(feature_cols)}")

# ── Step 5: Scale ────────────────────────────────────────────
print("\n📏 Scaling features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ── Step 6: Train/Test split ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\n✂️  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# Save global test set
np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

# ── Step 7: Non-IID split across 5 clients ───────────────────
# Each client gets DIFFERENT attack types (realistic!)
print(f"\n🏢 Creating Non-IID split across {NUM_CLIENTS} clients...")

classes = list(le.classes_)
num_classes = len(classes)

# Assign different attack types to different clients
client_class_map = {}
for i in range(NUM_CLIENTS):
    # Each client gets some classes more than others
    client_class_map[i] = []

for cls_idx in range(num_classes):
    # Primary client gets 60% of this class
    primary = cls_idx % NUM_CLIENTS
    client_class_map[primary].append(cls_idx)

print("\n   Client → Attack types assigned:")
for i, cls_list in client_class_map.items():
    names = [classes[c] for c in cls_list]
    print(f"   Client {i}: {names}")

# Split data accordingly
for client_id in range(NUM_CLIENTS):
    assigned_classes = client_class_map[client_id]
    
    # Get indices for assigned classes (60%) + random sample of others (40%)
    primary_mask = np.isin(y_train, assigned_classes)
    other_mask   = ~primary_mask
    
    primary_idx = np.where(primary_mask)[0]
    other_idx   = np.where(other_mask)[0]
    
    # Sample 40% from other classes
    n_other = min(len(other_idx), int(0.4 * len(primary_idx)))
    if n_other > 0:
        other_sample = np.random.choice(other_idx, n_other, replace=False)
        client_idx   = np.concatenate([primary_idx, other_sample])
    else:
        client_idx = primary_idx
    
    np.random.shuffle(client_idx)
    
    np.save(f"{OUTPUT_DIR}/client_{client_id}/X_train.npy", X_train[client_idx])
    np.save(f"{OUTPUT_DIR}/client_{client_id}/y_train.npy", y_train[client_idx])
    print(f"   Client {client_id}: {len(client_idx):,} samples saved ✅")

# ── Step 8: Save artifacts ───────────────────────────────────
with open(f"{OUTPUT_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(f"{OUTPUT_DIR}/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open(f"{OUTPUT_DIR}/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
with open(f"{OUTPUT_DIR}/classes.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(le.classes_))

print("\n" + "="*60)
print("✅ PREPROCESSING COMPLETE!")
print(f"   Saved to: {OUTPUT_DIR}")
print("   Next: python models/lstm_autoencoder.py")
print("="*60)