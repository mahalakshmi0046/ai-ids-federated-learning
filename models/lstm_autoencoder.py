# ============================================================
# LSTM AUTOENCODER — The Real AI Core
# This is what makes your project "AI-Powered"
#
# HOW IT WORKS:
# - Trained ONLY on normal (BENIGN) traffic
# - Learns what "normal" looks like
# - When it sees an attack, it can't reconstruct it well
# - High reconstruction error = ATTACK DETECTED
# - This detects ZERO-DAY attacks too! (never seen before)
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score

print("="*60)
print("  LSTM Autoencoder — Anomaly Detection Model")
print("="*60)

DATA_DIR   = r"C:\ids_project\data"
MODEL_DIR  = r"C:\ids_project\models"
RESULT_DIR = r"C:\ids_project\results"

os.makedirs(RESULT_DIR, exist_ok=True)

SEQUENCE_LEN = 10     # Look at 10 packets at a time (sequence)
HIDDEN_SIZE  = 64     # LSTM hidden units
LATENT_SIZE  = 32     # Compressed representation size
EPOCHS       = 30     # Training epochs
BATCH_SIZE   = 256    # Batch size
LR           = 0.001  # Learning rate
THRESHOLD_PERCENTILE = 95  # Flag top 5% reconstruction errors as attacks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Device: {device}")

# ── Load Data ────────────────────────────────────────────────
print("\n📂 Loading preprocessed data...")
X_test  = np.load(f"{DATA_DIR}/X_test.npy")
y_test  = np.load(f"{DATA_DIR}/y_test.npy")

with open(f"{DATA_DIR}/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load all client training data
all_X = []
for i in range(5):
    X = np.load(f"{DATA_DIR}/client_{i}/X_train.npy")
    all_X.append(X)
X_train = np.vstack(all_X)
y_train_combined = np.concatenate([
    np.load(f"{DATA_DIR}/client_{i}/y_train.npy") for i in range(5)
])

print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# Get BENIGN class index
classes = list(le.classes_)
benign_idx = classes.index('BENIGN')
print(f"   BENIGN index: {benign_idx}")

# ── Filter BENIGN only for training ──────────────────────────
# Autoencoder trains ONLY on normal traffic!
benign_mask = y_train_combined == benign_idx
X_benign    = X_train[benign_mask]
print(f"   BENIGN training samples: {len(X_benign):,}")

input_size = X_train.shape[1]
print(f"   Input features: {input_size}")

# ── Create Sequences ─────────────────────────────────────────
def create_sequences(X, seq_len):
    """
    Convert flat feature vectors into sequences.
    LSTM needs sequences — we group seq_len packets together.
    
    Example: seq_len=10 means model looks at 10 packets at once
    to understand traffic PATTERNS over time, not just one packet.
    """
    sequences = []
    for i in range(0, len(X) - seq_len, seq_len):
        sequences.append(X[i:i+seq_len])
    return np.array(sequences)

print(f"\n🔄 Creating sequences (length={SEQUENCE_LEN})...")
X_benign_seq = create_sequences(X_benign, SEQUENCE_LEN)
X_test_seq   = create_sequences(X_test,   SEQUENCE_LEN)
y_test_seq   = y_test[SEQUENCE_LEN:len(X_test_seq)*SEQUENCE_LEN+SEQUENCE_LEN:SEQUENCE_LEN]
y_test_seq   = y_test[:len(X_test_seq)]

print(f"   Benign sequences: {len(X_benign_seq):,}")
print(f"   Test sequences:   {len(X_test_seq):,}")

# ── LSTM Autoencoder Model ───────────────────────────────────
class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for Anomaly Detection.
    
    ENCODER: Compresses normal traffic into a small representation
    DECODER: Reconstructs the traffic from that representation
    
    Normal traffic → small reconstruction error (model knows it)
    Attack traffic → large reconstruction error (model never saw it)
    
    This is UNSUPERVISED anomaly detection — no attack labels needed!
    """
    
    def __init__(self, input_size, hidden_size, latent_size, seq_len):
        super(LSTMAutoencoder, self).__init__()
        
        self.seq_len    = seq_len
        self.hidden_size = hidden_size
        
        # ENCODER — compresses input
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.encoder_fc = nn.Linear(hidden_size, latent_size)
        
        # DECODER — reconstructs from compressed form
        self.decoder_fc   = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.output_layer = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        # Encode
        enc_out, (h_n, _) = self.encoder_lstm(x)
        latent = self.encoder_fc(h_n[-1])
        
        # Decode
        dec_input = self.decoder_fc(latent)
        dec_input = dec_input.unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder_lstm(dec_input)
        reconstruction = self.output_layer(dec_out)
        
        return reconstruction, latent
    
    def reconstruction_error(self, x):
        """Calculate how different input is from reconstruction."""
        recon, _ = self.forward(x)
        error = torch.mean((x - recon) ** 2, dim=(1, 2))
        return error


# ── Train ────────────────────────────────────────────────────
if __name__ == "__main__":
    model = LSTMAutoencoder(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        seq_len=SEQUENCE_LEN
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"\n🧠 Training LSTM Autoencoder...")
    print(f"   Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")

    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_benign_seq)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    train_losses = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1:3d}/{EPOCHS} — Loss: {avg_loss:.6f}")

    print("   ✅ Training complete!")

    # ── Set Anomaly Threshold ────────────────────────────────────
    print(f"\n📏 Setting anomaly threshold...")
    model.eval()
    with torch.no_grad():
        benign_tensor = torch.FloatTensor(X_benign_seq[:5000]).to(device)
        errors = model.reconstruction_error(benign_tensor).cpu().numpy()

    threshold = np.percentile(errors, THRESHOLD_PERCENTILE)
    print(f"   Threshold (p{THRESHOLD_PERCENTILE}): {threshold:.6f}")
    print(f"   Meaning: reconstruction error > {threshold:.4f} = ATTACK")

    # ── Evaluate ─────────────────────────────────────────────────
    print(f"\n📊 Evaluating on test set...")
    model.eval()
    all_errors = []

    with torch.no_grad():
        batch_size = 512
        for i in range(0, len(X_test_seq), batch_size):
            batch = torch.FloatTensor(X_test_seq[i:i+batch_size]).to(device)
            errors_batch = model.reconstruction_error(batch).cpu().numpy()
            all_errors.extend(errors_batch)

    all_errors = np.array(all_errors)

    # Predict: high error = attack (1), low error = benign (0)
    y_pred_binary = (all_errors > threshold).astype(int)
    y_true_binary = (y_test_seq != benign_idx).astype(int)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    acc  = accuracy_score(y_true_binary, y_pred_binary)
    f1   = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    rec  = recall_score(y_true_binary, y_pred_binary, zero_division=0)

    print(f"\n{'─'*40}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"{'─'*40}")

    # ── Save Everything ──────────────────────────────────────────
    print(f"\n💾 Saving model and artifacts...")

    torch.save(model.state_dict(), f"{MODEL_DIR}/lstm_autoencoder.pth")
    np.save(f"{MODEL_DIR}/threshold.npy", np.array([threshold]))
    np.save(f"{MODEL_DIR}/train_losses.npy", np.array(train_losses))
    np.save(f"{RESULT_DIR}/test_errors.npy", all_errors)
    np.save(f"{RESULT_DIR}/y_true_binary.npy", y_true_binary)
    np.save(f"{RESULT_DIR}/y_pred_binary.npy", y_pred_binary)

    # Save model config
    model_config = {
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,
        "latent_size": LATENT_SIZE,
        "seq_len": SEQUENCE_LEN,
        "threshold": threshold
    }
    with open(f"{MODEL_DIR}/model_config.pkl", "wb") as f:
        pickle.dump(model_config, f)

    # ── Plot Training Loss ───────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, color='#0EA5E9', linewidth=2)
    plt.title('LSTM Autoencoder Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss (MSE)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/training_loss.png", dpi=150)
    plt.close()

    # ── Plot Reconstruction Error Distribution ───────────────────
    plt.figure(figsize=(10, 5))
    benign_errors  = all_errors[y_true_binary == 0]
    attack_errors  = all_errors[y_true_binary == 1]

    plt.hist(benign_errors, bins=50, alpha=0.6, color='#10B981',
            label='Normal Traffic', density=True)
    plt.hist(attack_errors, bins=50, alpha=0.6, color='#EF4444',
            label='Attack Traffic', density=True)
    plt.axvline(threshold, color='#F59E0B', linewidth=2,
                linestyle='--', label=f'Threshold = {threshold:.4f}')
    plt.title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/error_distribution.png", dpi=150)
    plt.close()

    print(f"   ✅ Plots saved to {RESULT_DIR}/")

    print("\n" + "="*60)
    print(f"✅ LSTM AUTOENCODER COMPLETE!")
    print(f"   Accuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%")
    print(f"   Threshold: {threshold:.6f}")
    print(f"   Next: python models/xai_explainer.py")
    print("="*60)