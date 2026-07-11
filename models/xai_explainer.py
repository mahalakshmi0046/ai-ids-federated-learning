# ============================================================
# XAI MODULE — SHAP + LIME Explainability
# Answers: WHY did the model flag this as an attack?
# ============================================================

import numpy as np
import pickle
import os
import torch
import shap
import lime
from lime import lime_tabular
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("  XAI Module — SHAP + LIME Explainability")
print("="*60)

DATA_DIR   = r"C:\ids_project\data"
MODEL_DIR  = r"C:\ids_project\models"
RESULT_DIR = r"C:\ids_project\results"
os.makedirs(f"{RESULT_DIR}/xai", exist_ok=True)

# ── Load artifacts ───────────────────────────────────────────
print("\n📂 Loading model and data...")

with open(f"{MODEL_DIR}/model_config.pkl", "rb") as f:
    config = pickle.load(f)

with open(f"{DATA_DIR}/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open(f"{DATA_DIR}/feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

with open(f"{DATA_DIR}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_test = np.load(f"{DATA_DIR}/X_test.npy")
y_test = np.load(f"{DATA_DIR}/y_test.npy")

threshold   = config['threshold']
input_size  = config['input_size']
hidden_size = config['hidden_size']
latent_size = config['latent_size']
seq_len     = config['seq_len']

classes    = list(le.classes_)
benign_idx = classes.index('BENIGN')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Reload LSTM Autoencoder ──────────────────────────────────
import sys
sys.path.append(r"C:\ids_project\models")
from lstm_autoencoder import LSTMAutoencoder

model = LSTMAutoencoder(input_size, hidden_size, latent_size, seq_len).to(device)
model.load_state_dict(torch.load(
    f"{MODEL_DIR}/lstm_autoencoder.pth", map_location=device
))
model.eval()
print("   ✅ Model loaded!")

# ── Create flat scoring function for XAI ─────────────────────
# SHAP and LIME work on flat vectors, not sequences
# We reshape internally

def score_samples(X_flat):
    """
    Takes flat feature vectors → returns reconstruction error.
    Higher score = more likely an attack.
    """
    if len(X_flat.shape) == 1:
        X_flat = X_flat.reshape(1, -1)

    n = len(X_flat)
    # Pad to make sequences
    pad_needed = seq_len - (n % seq_len) if n % seq_len != 0 else 0
    if pad_needed > 0:
        X_flat = np.vstack([X_flat, np.zeros((pad_needed, X_flat.shape[1]))])

    # Create sequences
    sequences = []
    for i in range(0, len(X_flat) - seq_len + 1, seq_len):
        sequences.append(X_flat[i:i+seq_len])

    if len(sequences) == 0:
        seq = np.zeros((1, seq_len, X_flat.shape[1]))
        sequences = [seq[0]]

    X_seq = np.array(sequences)

    with torch.no_grad():
        t = torch.FloatTensor(X_seq).to(device)
        errors = model.reconstruction_error(t).cpu().numpy()

    # Return scores for each original sample
    scores = np.repeat(errors, seq_len)[:n + pad_needed][:n - pad_needed if pad_needed else n]
    if len(scores) == 0:
        scores = errors[:1]

    # Return as 2-class probability [normal, attack]
    attack_prob  = np.clip(scores / (threshold * 2), 0, 1)
    normal_prob  = 1 - attack_prob
    return np.column_stack([normal_prob, attack_prob])


# ── Get sample attack packets for explanation ────────────────
print("\n🔍 Selecting sample packets to explain...")

attack_mask = y_test != benign_idx
X_attacks   = X_test[attack_mask][:200]
y_attacks   = y_test[attack_mask][:200]

benign_mask = y_test == benign_idx
X_benign    = X_test[benign_mask][:200]

print(f"   Attack samples: {len(X_attacks)}")
print(f"   Benign samples: {len(X_benign)}")

# ── SHAP Explainability ──────────────────────────────────────
print("\n🔷 Running SHAP analysis...")
print("   (This takes 2-3 minutes...)")

# Use KernelExplainer (works with any model)
background = shap.kmeans(X_benign[:100], 10)  # 10 cluster centers as background

explainer_shap = shap.KernelExplainer(
    lambda x: score_samples(x)[:, 1],  # Return attack probability
    background
)

# Explain a small batch of attacks
X_explain = X_attacks[:20]
shap_values = explainer_shap.shap_values(X_explain, nsamples=100)

print("   ✅ SHAP values computed!")

# Plot 1: SHAP Summary Bar Plot (global feature importance)
plt.figure(figsize=(12, 8))
feature_importance = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(feature_importance)[-20:]  # Top 20 features
top_features = [feature_cols[i] if i < len(feature_cols) else f"Feature_{i}"
                for i in top_idx]
top_importance = feature_importance[top_idx]

colors = ['#EF4444' if v > np.median(top_importance) else '#0EA5E9'
          for v in top_importance]

plt.barh(range(len(top_features)), top_importance, color=colors)
plt.yticks(range(len(top_features)), top_features, fontsize=9)
plt.xlabel('Mean |SHAP Value| (Feature Importance)')
plt.title('SHAP — Global Feature Importance for Attack Detection',
          fontsize=13, fontweight='bold')
plt.axvline(np.median(top_importance), color='#F59E0B',
            linestyle='--', alpha=0.7, label='Median importance')
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/xai/shap_global_importance.png", dpi=150)
plt.close()
print("   ✅ Saved: shap_global_importance.png")

# Plot 2: SHAP for single attack packet
plt.figure(figsize=(12, 6))
single_shap = shap_values[0]
top_single_idx = np.argsort(np.abs(single_shap))[-15:]
top_single_feat = [feature_cols[i] if i < len(feature_cols)
                   else f"Feature_{i}" for i in top_single_idx]
top_single_vals = single_shap[top_single_idx]

colors_single = ['#EF4444' if v > 0 else '#10B981' for v in top_single_vals]
plt.barh(range(len(top_single_feat)), top_single_vals, color=colors_single)
plt.yticks(range(len(top_single_feat)), top_single_feat, fontsize=9)
plt.xlabel('SHAP Value (Red = pushes toward attack, Green = pushes toward normal)')
plt.title('SHAP — Why was THIS packet flagged as an attack?',
          fontsize=13, fontweight='bold')
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/xai/shap_single_explanation.png", dpi=150)
plt.close()
print("   ✅ Saved: shap_single_explanation.png")

# ── LIME Explainability ──────────────────────────────────────
print("\n🟡 Running LIME analysis...")

feature_names = [feature_cols[i] if i < len(feature_cols)
                 else f"Feature_{i}" for i in range(input_size)]

explainer_lime = lime_tabular.LimeTabularExplainer(
    X_benign,
    feature_names=feature_names,
    class_names=['Normal', 'Attack'],
    mode='classification',
    random_state=42
)

# Explain 3 different attack types
attack_types_explained = []
for i, (x, y_true) in enumerate(zip(X_attacks[:10], y_attacks[:10])):
    attack_name = classes[y_true]
    if attack_name not in attack_types_explained:
        attack_types_explained.append(attack_name)

        exp = explainer_lime.explain_instance(
            x,
            lambda X: score_samples(X),
            num_features=10,
            num_samples=200
        )

        # Plot LIME explanation
        fig = exp.as_pyplot_figure()
        fig.suptitle(f'LIME — Why "{attack_name}" was detected?',
                     fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        safe_name = attack_name.replace(" ", "_").replace("/", "_")
        plt.savefig(f"{RESULT_DIR}/xai/lime_{safe_name}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: lime_{safe_name}.png")

    if len(attack_types_explained) >= 3:
        break

# ── Save SHAP values for dashboard ──────────────────────────
np.save(f"{RESULT_DIR}/xai/shap_values.npy", shap_values)
np.save(f"{RESULT_DIR}/xai/feature_importance.npy", feature_importance)

# Save top features summary
top20_idx   = np.argsort(feature_importance)[-20:][::-1]
top20_names = [feature_cols[i] if i < len(feature_cols)
               else f"Feature_{i}" for i in top20_idx]
top20_vals  = feature_importance[top20_idx]

summary = {
    "top_features": top20_names,
    "importance_scores": top20_vals.tolist()
}
with open(f"{RESULT_DIR}/xai/feature_summary.pkl", "wb") as f:
    pickle.dump(summary, f)

print("\n" + "="*60)
print("✅ XAI MODULE COMPLETE!")
print(f"   SHAP plots saved to: {RESULT_DIR}/xai/")
print(f"   LIME plots saved to: {RESULT_DIR}/xai/")
print("   Next: python models/fl_server.py")
print("="*60)