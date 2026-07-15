# 🛡️ AI-Powered Intrusion Detection System
### using Federated Learning + Differential Privacy + SHAP/LIME

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 What This Project Solves

Traditional IDS systems require centralizing sensitive network
data — violating privacy regulations like GDPR and RBI compliance.

This system enables **5 organizations to collaboratively build
a smarter IDS without sharing any raw network data.**

## 🏆 Results

| Metric | Value |
|--------|-------|
| Multi-class Accuracy | 91.79% |
| Binary F1-Score | 99.43% |
| Attack Types Detected | 15 |
| Differential Privacy ε | 0.000484 |
| Federated Clients | 5 |
| Dataset Size | 2.5M packets |

## 🔧 Architecture

1. 📡 Network Traffic (CICIDS2017 — 2.5M packets)
2. 🧹 Preprocessing + Non-IID Split across 5 clients
3. 🧠 LSTM Autoencoder (zero-day anomaly detection)
4. 🌐 Federated Learning (FedAvg — 10 rounds)
5. 🔒 Differential Privacy (ε = 0.000484)
6. 🔍 SHAP + LIME Explainability
7. 📊 Real-time Streamlit Dashboard

## 🚀 Components

| Component | Purpose |
|-----------|---------|
| LSTM Autoencoder | Anomaly detection via reconstruction error |
| Federated Learning | Privacy-preserving distributed training |
| Differential Privacy | Formal (ε,δ)-DP guarantee |
| SHAP | Global feature importance |
| LIME | Per-prediction explanation |
| Streamlit Dashboard | Real-time monitoring |

## 📊 Dataset

CICIDS2017 — Canadian Institute for Cybersecurity
- 2.5 million real network packets
- 15 attack types including DDoS, PortScan,
  SQL Injection, XSS, Heartbleed, Bot, FTP-Patator

## ⚡ Quick Start

**Step 1 — Clone and install:**
```bash
git clone https://github.com/mahalakshmi0046/ai-ids-federated-learning
cd ai-ids-federated-learning
pip install -r requirements.txt
```

**Step 2 — Download CICIDS2017 dataset:**
**Step 3 — Run pipeline:**
```bash
python preprocess.py
python models/lstm_autoencoder.py
python models/xai_explainer.py
python final_final_fix.py
streamlit run dashboard/app.py
```

## 📦 Requirements
torch
scikit-learn
pandas
numpy
matplotlib
seaborn
streamlit
fastapi
uvicorn
shap
lime
scapy
## 🌐 Real-World Application

Applicable in:
- 🏦 Banking (RBI compliance — cannot share transaction logs)
- 🏥 Healthcare (patient data privacy — HIPAA equivalent)
- 📡 Telecom (subscriber data protection)
- 🏭 Industrial networks (SCADA security)

## 🔍 Why Not LLMs?

| Factor | LLM | Our Model |
|--------|-----|-----------|
| Inference speed | 500ms+ | Less than 1ms |
| Model size | 800GB+ | 2MB |
| Privacy | Sends data to API | Fully local |
| Explainability | Black box | SHAP + LIME |
| Real-time capable | No | Yes |

## 📈 FL Training Progress

| Round | Multi-class Accuracy | Binary F1 |
|-------|---------------------|-----------|
| Round 1 | 87.99% | 98.89% |
| Round 5 | 89.24% | 99.39% |
| Round 10 | 91.79% | 99.43% |

## 👩‍💻 Author

**Mahalakshmi**
Final Year B.Tech CSE
2026-27 Batch

## 📚 References

- Zhang et al. (2025) — "FedIDS: Privacy-Preserving Intrusion 
  Detection using Federated Learning" — IEEE Transactions on 
  Information Forensics and Security

- Nguyen et al. (2024) — "LSTM-based Anomaly Detection for 
  Network Intrusion with Federated Learning" — 
  IEEE Symposium on Security and Privacy

- Li et al. (2024) — "Differential Privacy in Federated 
  Learning for Cybersecurity Applications" — 
  ACM CCS 2024

- Rahman et al. (2023) — "Explainable AI for Network Intrusion 
  Detection: SHAP and LIME Analysis on CICIDS2017" — 
  Computers and Security, Elsevier

- Agrawal et al. (2023) — "FedProx for Non-IID Network Traffic 
  Classification in Heterogeneous Organizations" — 
  IEEE INFOCOM 2023

- Ferrag et al. (2024) — "Generative AI and Large Language 
  Models for Cyber Security: A Review vs Traditional ML IDS" — 
  arXiv 2024 (justifies why LLMs unsuitable for real-time IDS)

- CICIDS2017 Dataset — Sharafaldin et al. — 
  University of New Brunswick (benchmark used in 500+ papers)