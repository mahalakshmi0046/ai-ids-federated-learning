# ============================================================
# LIVE NETWORK CAPTURE — Real Network Integration
# Captures live packets from your network interface
# and runs them through the IDS model in real time
# ============================================================

from scapy.all import sniff, IP, TCP, UDP
import numpy as np
import torch
import pickle
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = r"C:\ids_project\models"
DATA_DIR  = r"C:\ids_project\data"

print("="*60)
print("  LIVE NETWORK IDS — Real Packet Capture")
print("="*60)

# Load model artifacts
with open(f"{MODEL_DIR}/model_config.pkl", "rb") as f:
    config = pickle.load(f)

with open(f"{DATA_DIR}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(f"{DATA_DIR}/classes.txt", encoding="utf-8") as f:
    classes = f.read().strip().split("\n")

threshold = config['threshold']
device    = torch.device("cpu")

# Load model
import sys
sys.path.append(MODEL_DIR)
from lstm_autoencoder import LSTMAutoencoder

model = LSTMAutoencoder(
    config['input_size'],
    config['hidden_size'],
    config['latent_size'],
    config['seq_len']
).to(device)

# Load FL trained model if exists, else use base model
import os
model_path = f"{MODEL_DIR}/fl_global_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    print("✅ Loaded FL global model")
else:
    model.load_state_dict(
        torch.load(f"{MODEL_DIR}/lstm_autoencoder.pth",
                   map_location=device)
    )
    print("✅ Loaded base LSTM model")

model.eval()

# Packet buffer — collect seq_len packets then classify
packet_buffer = []
seq_len       = config['seq_len']
input_size    = config['input_size']

def extract_features(packet):
    """
    Extract numerical features from a live packet.
    Maps to the same feature space as CICIDS2017.
    """
    features = np.zeros(input_size)
    try:
        if IP in packet:
            features[0] = len(packet)                    # Packet length
            features[1] = packet[IP].ttl                 # TTL
            features[2] = packet[IP].proto               # Protocol

            if TCP in packet:
                features[3] = packet[TCP].sport          # Source port
                features[4] = packet[TCP].dport          # Dest port
                features[5] = packet[TCP].flags          # TCP flags
                features[6] = packet[TCP].window         # Window size
                features[7] = len(packet[TCP].payload)   # Payload length

            elif UDP in packet:
                features[3] = packet[UDP].sport
                features[4] = packet[UDP].dport
                features[7] = len(packet[UDP].payload)

            features[8]  = time.time() % 1000           # Timestamp mod
            features[9]  = int(packet[IP].flags)        # IP flags
    except Exception:
        pass
    return features

def classify_packet_buffer():
    """Run the LSTM Autoencoder on buffered packets."""
    global packet_buffer

    if len(packet_buffer) < seq_len:
        return

    # Take seq_len packets
    sequence = np.array(packet_buffer[:seq_len])
    packet_buffer = packet_buffer[seq_len:]  # Remove used packets

    # Scale features
    try:
        sequence_scaled = scaler.transform(sequence)
    except Exception:
        sequence_scaled = sequence

    # Run through model
    seq_tensor = torch.FloatTensor(
        sequence_scaled.reshape(1, seq_len, input_size)
    ).to(device)

    with torch.no_grad():
        error = model.reconstruction_error(seq_tensor).item()

    # Classify
    is_attack = error > threshold
    status    = "🔴 ATTACK" if is_attack else "🟢 NORMAL"
    confidence = min(99.9, (error / threshold) * 50 + 50) \
                 if is_attack else \
                 min(99.9, (1 - error/threshold) * 50 + 50)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
          f"{status} | "
          f"Recon Error: {error:.4f} | "
          f"Threshold: {threshold:.4f} | "
          f"Confidence: {confidence:.1f}%")

    if is_attack:
        print(f"  ⚠️  ALERT: Anomalous traffic detected!")
        print(f"  Reconstruction error {error:.4f} "
              f"is {error/threshold:.1f}x above threshold")

def process_packet(packet):
    """Called for every captured packet."""
    features = extract_features(packet)
    packet_buffer.append(features)
    classify_packet_buffer()

# Start capturing
print(f"\n🌐 Starting live packet capture...")
print(f"   Threshold    : {threshold:.4f}")
print(f"   Sequence len : {seq_len}")
print(f"   Input size   : {input_size}")
print(f"\n   Capturing from ALL network interfaces...")
print(f"   Press Ctrl+C to stop\n")
print("-" * 60)

try:
    sniff(
        prn=process_packet,
        store=False,        # Don't store packets in memory
        filter="ip",        # Only capture IP packets
    )
except KeyboardInterrupt:
    print("\n\n✅ Capture stopped.")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("   Try running as Administrator for packet capture")
    print("   Right-click VS Code → Run as Administrator")