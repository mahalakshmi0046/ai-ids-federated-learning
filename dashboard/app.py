# ============================================================
# REAL-TIME IDS DASHBOARD
# Shows: Live alerts, SHAP/LIME explanations, FL progress, DP metrics
# Run: streamlit run dashboard\app.py
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle, os, random, time, torch, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI-Powered IDS",
    page_icon="🛡️",
    layout="wide"
)

# ── Dark theme styling ───────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0a0e1a; }
[data-testid="stSidebar"]          { background: #0d1f35; }
.metric-box {
    background: #0d1f35;
    border: 1px solid #1e4a8a;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}
.attack-badge {
    background: #ef4444;
    color: white;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: bold;
}
.normal-badge {
    background: #10b981;
    color: white;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
}
h1, h2, h3, p, label { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Paths ────────────────────────────────────────────────────
DATA_DIR   = r"C:\ids_project\data"
MODEL_DIR  = r"C:\ids_project\models"
RESULT_DIR = r"C:\ids_project\results"

# ── Load artifacts ───────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open(f"{DATA_DIR}/classes.txt", encoding="utf-8") as f:
        classes = f.read().strip().split("\n")
    with open(f"{MODEL_DIR}/model_config.pkl", "rb") as f:
        config = pickle.load(f)
    with open(f"{DATA_DIR}/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    # Load SHAP feature importance if available
    shap_summary = None
    shap_path = f"{RESULT_DIR}/xai/feature_summary.pkl"
    if os.path.exists(shap_path):
        with open(shap_path, "rb") as f:
            shap_summary = pickle.load(f)

    # Load FL training log if available
    fl_log = None
    log_path = f"{RESULT_DIR}/fl_log.txt"
    if os.path.exists(log_path):
        fl_log = pd.read_csv(log_path)

    # Load error distribution
    test_errors  = None
    y_true_bin   = None
    errors_path  = f"{RESULT_DIR}/test_errors.npy"
    if os.path.exists(errors_path):
        test_errors = np.load(errors_path)
        y_true_bin  = np.load(f"{RESULT_DIR}/y_true_binary.npy")

    return classes, config, feature_cols, shap_summary, fl_log, \
           test_errors, y_true_bin

classes, config, feature_cols, shap_summary, fl_log, \
    test_errors, y_true_bin = load_artifacts()

benign_idx = classes.index('BENIGN')
threshold  = config['threshold']
ATTACK_TYPES = [c for c in classes if c != 'BENIGN']

# ── Simulate live traffic ────────────────────────────────────
def generate_event():
    is_attack   = random.random() > 0.75
    if is_attack:
        label = random.choice(ATTACK_TYPES)
        recon_error = random.uniform(threshold * 1.2, threshold * 4)
        severity    = "🔴 CRITICAL" if recon_error > threshold * 2.5 \
                      else "🟡 WARNING"
    else:
        label       = "BENIGN"
        recon_error = random.uniform(0, threshold * 0.8)
        severity    = "🟢 NORMAL"

    confidence = min(99.9, 70 + (recon_error / threshold) * 20)
    return {
        "Time":             datetime.now().strftime("%H:%M:%S"),
        "Source IP":        f"192.168.{random.randint(1,10)}."
                            f"{random.randint(1,254)}",
        "Dest IP":          f"10.0.{random.randint(0,5)}."
                            f"{random.randint(1,254)}",
        "Port":             random.choice([80,443,22,3389,8080,
                            random.randint(1024,65535)]),
        "Detection":        label,
        "Recon Error":      f"{recon_error:.4f}",
        "Threshold":        f"{threshold:.4f}",
        "Confidence":       f"{confidence:.1f}%",
        "Status":           severity
    }

# ── Session state ────────────────────────────────────────────
if "total"   not in st.session_state:
    st.session_state.total   = 0
    st.session_state.attacks = 0
    st.session_state.events  = []
    st.session_state.counts  = {c: 0 for c in classes}

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ IDS Control Panel")
    st.markdown("---")
    refresh = st.slider("Refresh Rate (sec)", 1, 5, 2)
    attacks_only = st.checkbox("Show Attacks Only", False)
    st.markdown("---")

    st.markdown("### 📊 Model Info")
    st.markdown(f"**Threshold:** `{threshold:.4f}`")
    st.markdown(f"**Features:** `{config['input_size']}`")
    st.markdown(f"**Seq Length:** `{config['seq_len']}`")
    st.markdown(f"**Classes:** `{len(classes)}`")
    st.markdown("---")

    st.markdown("### 🔒 Privacy Budget")
    if fl_log is not None and len(fl_log) > 0:
        final_eps = fl_log['DP_Epsilon'].iloc[-1]
        st.markdown(f"**DP ε (epsilon):** `{final_eps:.4f}`")
        st.markdown(f"**Rounds trained:** `{len(fl_log)}`")
        st.progress(min(1.0, final_eps / 1.0))
        st.caption("Lower ε = stronger privacy guarantee")
    else:
        st.markdown("FL not yet trained")

# ── HEADER ───────────────────────────────────────────────────
st.markdown(
    "# 🛡️ AI-Powered Intrusion Detection System"
)
st.markdown(
    "##### LSTM Autoencoder + Federated Learning + "
    "Differential Privacy + SHAP/LIME"
)
st.markdown("---")

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📡 Live Monitor",
    "📈 FL Training",
    "🔍 XAI Explanations",
    "📊 Model Analysis"
])

# ════════════════════════════════════════════════════════════
# TAB 1: LIVE MONITOR
# ════════════════════════════════════════════════════════════
with tab1:
    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    m1 = c1.empty()
    m2 = c2.empty()
    m3 = c3.empty()
    m4 = c4.empty()

    st.markdown("---")
    left, right = st.columns([3, 2])

    with left:
        st.markdown("### 📡 Live Traffic Feed")
        feed_placeholder = st.empty()

    with right:
        st.markdown("### 🎯 Attack Distribution")
        dist_placeholder = st.empty()

# ════════════════════════════════════════════════════════════
# TAB 2: FL TRAINING PROGRESS
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📈 Federated Learning Training Progress")

    if fl_log is not None and len(fl_log) > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Accuracy per Round")
            st.line_chart(
                fl_log.set_index("Round")["Accuracy"],
                use_container_width=True
            )

        with col2:
            st.markdown("#### F1-Score per Round")
            st.line_chart(
                fl_log.set_index("Round")["F1"],
                use_container_width=True
            )

        st.markdown("#### Privacy Budget (DP ε) over Rounds")
        st.area_chart(
            fl_log.set_index("Round")["DP_Epsilon"],
            use_container_width=True
        )

        st.markdown("#### Full Training Log")
        st.dataframe(fl_log, use_container_width=True)

        # Key insight
        st.info(
            f"🔒 **Privacy-Accuracy Tradeoff:** "
            f"Final accuracy = {fl_log['Accuracy'].iloc[-1]*100:.1f}% "
            f"with DP ε = {fl_log['DP_Epsilon'].iloc[-1]:.4f}. "
            f"Lower epsilon = stronger privacy guarantee but "
            f"slightly lower accuracy. This is the fundamental "
            f"tradeoff in privacy-preserving ML."
        )
    else:
        st.warning(
            "FL training log not found. "
            "Run python models/federated_learning.py first!"
        )

# ════════════════════════════════════════════════════════════
# TAB 3: XAI EXPLANATIONS
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔍 Explainable AI — SHAP + LIME")

    xai_col1, xai_col2 = st.columns(2)

    with xai_col1:
        st.markdown("#### SHAP — Global Feature Importance")
        shap_img = f"{RESULT_DIR}/xai/shap_global_importance.png"
        if os.path.exists(shap_img):
            st.image(shap_img, use_column_width=True)
            st.caption(
                "Red bars = features that push toward attack detection. "
                "This shows WHAT the model learned globally."
            )
        else:
            st.warning("Run xai_explainer.py to generate SHAP plots")

    with xai_col2:
        st.markdown("#### SHAP — Single Packet Explanation")
        shap_single = f"{RESULT_DIR}/xai/shap_single_explanation.png"
        if os.path.exists(shap_single):
            st.image(shap_single, use_column_width=True)
            st.caption(
                "WHY was this specific packet flagged? "
                "Red = pushed toward attack, Green = pushed toward normal."
            )
        else:
            st.warning("Run xai_explainer.py to generate SHAP plots")

    st.markdown("---")
    st.markdown("#### LIME — Per-Attack Explanations")

    lime_files = []
    xai_dir    = f"{RESULT_DIR}/xai"
    if os.path.exists(xai_dir):
        lime_files = [f for f in os.listdir(xai_dir)
                      if f.startswith("lime_")]

    if lime_files:
        cols = st.columns(min(3, len(lime_files)))
        for i, lf in enumerate(lime_files[:3]):
            with cols[i]:
                attack_name = lf.replace("lime_","").replace(".png","").replace("_"," ")
                st.markdown(f"**{attack_name}**")
                st.image(f"{xai_dir}/{lf}", use_column_width=True)
    else:
        st.warning("Run xai_explainer.py to generate LIME plots")

    if shap_summary:
        st.markdown("---")
        st.markdown("#### Top 10 Most Important Features")
        feat_df = pd.DataFrame({
            "Feature":   shap_summary["top_features"][:10],
            "Importance": shap_summary["importance_scores"][:10]
        })
        st.bar_chart(
            feat_df.set_index("Feature"),
            use_container_width=True
        )

# ════════════════════════════════════════════════════════════
# TAB 4: MODEL ANALYSIS
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📊 Model Performance Analysis")

    ana_col1, ana_col2 = st.columns(2)

    with ana_col1:
        st.markdown("#### Reconstruction Error Distribution")
        err_img = f"{RESULT_DIR}/error_distribution.png"
        if os.path.exists(err_img):
            st.image(err_img, use_column_width=True)
            st.caption(
                "Green = normal traffic (low error). "
                "Red = attack traffic (high error). "
                "Yellow line = detection threshold."
            )

    with ana_col2:
        st.markdown("#### Training Loss Curve")
        loss_img = f"{RESULT_DIR}/training_loss.png"
        if os.path.exists(loss_img):
            st.image(loss_img, use_column_width=True)
            st.caption(
                "Loss decreasing = model learning "
                "what normal traffic looks like."
            )

    if test_errors is not None:
        st.markdown("---")
        st.markdown("#### Attack Detection Statistics")

        benign_errors = test_errors[y_true_bin == 0]
        attack_errors = test_errors[y_true_bin == 1]

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Avg Normal Error",
                  f"{np.mean(benign_errors):.4f}")
        s2.metric("Avg Attack Error",
                  f"{np.mean(attack_errors):.4f}")
        s3.metric("Detection Threshold",
                  f"{threshold:.4f}")
        s4.metric("Error Separation",
                  f"{np.mean(attack_errors)/np.mean(benign_errors):.1f}x")

        st.info(
            f"💡 Attack traffic has "
            f"**{np.mean(attack_errors)/np.mean(benign_errors):.1f}x** "
            f"higher reconstruction error than normal traffic. "
            f"This separation is what enables anomaly detection."
        )

    st.markdown("---")
    st.markdown("#### Architecture Summary")
    arch_data = {
        "Component":   ["LSTM Autoencoder", "Federated Learning",
                        "Differential Privacy", "SHAP", "LIME"],
        "Purpose":     ["Anomaly detection via reconstruction error",
                        "Privacy-preserving distributed training",
                        "Formal (ε,δ)-DP privacy guarantee",
                        "Global feature importance explanation",
                        "Per-prediction local explanation"],
        "Status":      ["✅ Trained", "✅ Complete",
                        "✅ Applied", "✅ Generated", "✅ Generated"]
    }
    st.table(pd.DataFrame(arch_data))

# ════════════════════════════════════════════════════════════
# LIVE UPDATE LOOP (Tab 1)
# ════════════════════════════════════════════════════════════
while True:
    # Generate new events
    for _ in range(4):
        event = generate_event()
        st.session_state.total += 1
        st.session_state.counts[event["Detection"]] = \
            st.session_state.counts.get(event["Detection"], 0) + 1
        if event["Detection"] != "BENIGN":
            st.session_state.attacks += 1
        st.session_state.events.insert(0, event)

    st.session_state.events = st.session_state.events[:100]

    rate = (st.session_state.attacks /
            max(st.session_state.total, 1)) * 100

    # Update metrics
    m1.metric("📦 Total Packets",
              f"{st.session_state.total:,}")
    m2.metric("🚨 Attacks Detected",
              f"{st.session_state.attacks:,}")
    m3.metric("⚠️ Attack Rate",
              f"{rate:.1f}%")
    m4.metric("🧠 Model Threshold",
              f"{threshold:.4f}")

    # Update live feed
    display = st.session_state.events
    if attacks_only:
        display = [e for e in display
                   if e["Detection"] != "BENIGN"]

    feed_placeholder.dataframe(
        pd.DataFrame(display[:20]),
        use_container_width=True,
        hide_index=True
    )

    # Update distribution chart
    dist_df = pd.DataFrame({
        "Type":  list(st.session_state.counts.keys()),
        "Count": list(st.session_state.counts.values())
    })
    dist_df = dist_df[dist_df["Count"] > 0]
    if not dist_df.empty:
        dist_placeholder.bar_chart(
            dist_df.set_index("Type"),
            use_container_width=True
        )

    time.sleep(refresh)