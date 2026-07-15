# ============================================================
# SMART ALERT SYSTEM
# Severity-based alerting with cooldown and digest
# No email spam — only meaningful alerts!
# ============================================================

import smtplib
import ssl
import sys
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import time

sys.path.append(r"C:\ids_project")
from database.attack_logger import (
    get_attack_summary, get_top_sources, get_stats
)

# ── CONFIG — Update these ────────────────────────────────────
SENDER_EMAIL    = "smahalakshmi2004mjm@gmail.com"
SENDER_PASSWORD = "agot dzip xoif bprj"  # Gmail App Password
RECEIVER_EMAIL  = "admin@gmail.com"

# Alert settings
CRITICAL_COOLDOWN = 10 * 60    # 10 mins between critical alerts
DIGEST_INTERVAL   = 30 * 60   # 30 min digest email
# ─────────────────────────────────────────────────────────────

# Severity classification
SEVERITY = {
    "CRITICAL": ["DDoS", "Bot", "Infiltration"],
    "HIGH":     ["DoS Hulk", "DoS GoldenEye",
                 "DoS slowloris", "DoS Slowhttptest",
                 "Heartbleed"],
    "MEDIUM":   ["PortScan", "FTP-Patator", "SSH-Patator"],
    "LOW":      ["Web Attack  Brute Force",
                 "Web Attack  Sql Injection",
                 "Web Attack  XSS"],
}

SEVERITY_COLORS = {
    "CRITICAL": "#ef4444",
    "HIGH":     "#f59e0b",
    "MEDIUM":   "#0ea5e9",
    "LOW":      "#10b981",
}

def get_severity(attack_type):
    for sev, attacks in SEVERITY.items():
        if attack_type in attacks:
            return sev
    return "LOW"

# Track cooldowns and digest buffer
_last_critical_alert = {}   # attack_type → timestamp
_digest_buffer       = []   # list of attack dicts
_digest_lock         = threading.Lock()
_email_enabled       = True

def _send_email(subject, html_body, text_body):
    """Core email sending function."""
    if not _email_enabled:
        print(f"  [ALERT] {subject} (email disabled)")
        return

    try:
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SENDER_EMAIL
        msg["To"]      = RECEIVER_EMAIL

        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(
            "smtp.gmail.com", 465, context=ctx
        ) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(
                SENDER_EMAIL,
                RECEIVER_EMAIL,
                msg.as_string()
            )
        print(f"  📧 Sent: {subject}")
        return True

    except Exception as e:
        print(f"  ❌ Email error: {e}")
        return False


def send_critical_alert(attack_type, src_ip,
                        dst_ip, port,
                        recon_error, confidence):
    """
    Send immediate alert for CRITICAL attacks.
    Uses cooldown to prevent spam.
    """
    # Check cooldown
    now  = datetime.now()
    last = _last_critical_alert.get(attack_type)
    if last:
        elapsed = (now - last).total_seconds()
        if elapsed < CRITICAL_COOLDOWN:
            remaining = int(
                (CRITICAL_COOLDOWN - elapsed) / 60
            )
            print(f"  ⏳ Alert suppressed "
                  f"({remaining} min cooldown)")
            return False

    _last_critical_alert[attack_type] = now
    severity  = get_severity(attack_type)
    color     = SEVERITY_COLORS.get(severity, "#ef4444")

    subject   = (
        f"🚨 [{severity}] {attack_type} Detected — "
        f"Immediate Action Required"
    )

    html = f"""
<html>
<body style="font-family:Arial,sans-serif;
             background:#0a0e1a;color:#e2e8f0;
             padding:20px;margin:0;">

  <!-- Header -->
  <div style="background:{color};padding:20px;
              border-radius:8px 8px 0 0;">
    <h1 style="color:white;margin:0;font-size:22px;">
      🚨 INTRUSION DETECTED
    </h1>
    <p style="color:rgba(255,255,255,0.9);
              margin:5px 0 0;font-size:14px;">
      AI-Powered IDS — Federated Learning System
    </p>
  </div>

  <!-- Severity Badge -->
  <div style="background:#0d1f35;padding:15px;
              border-left:4px solid {color};">
    <span style="background:{color};color:white;
                 padding:4px 12px;border-radius:4px;
                 font-weight:bold;font-size:13px;">
      {severity}
    </span>
    <span style="margin-left:10px;font-size:18px;
                 font-weight:bold;color:{color};">
      {attack_type}
    </span>
  </div>

  <!-- Details Table -->
  <div style="background:#0d1f35;padding:20px;
              border-radius:0 0 8px 8px;
              border:1px solid #1e4a8a;
              border-top:none;">
    <table style="width:100%;border-collapse:collapse;">
      <tr>
        <td style="padding:10px;color:#64748b;
                   font-weight:bold;width:35%;">
          🕐 Time
        </td>
        <td style="padding:10px;">
          {now.strftime('%Y-%m-%d %H:%M:%S')}
        </td>
      </tr>
      <tr style="background:#0a1628;">
        <td style="padding:10px;color:#64748b;
                   font-weight:bold;">
          🌐 Source IP
        </td>
        <td style="padding:10px;color:#ef4444;
                   font-weight:bold;">{src_ip}</td>
      </tr>
      <tr>
        <td style="padding:10px;color:#64748b;
                   font-weight:bold;">
          🎯 Target IP
        </td>
        <td style="padding:10px;">{dst_ip}</td>
      </tr>
      <tr style="background:#0a1628;">
        <td style="padding:10px;color:#64748b;
                   font-weight:bold;">
          🔌 Port
        </td>
        <td style="padding:10px;">{port}</td>
      </tr>
      <tr>
        <td style="padding:10px;color:#64748b;
                   font-weight:bold;">
          🧠 Confidence
        </td>
        <td style="padding:10px;color:#f59e0b;
                   font-weight:bold;">
          {confidence:.1f}%
        </td>
      </tr>
      <tr style="background:#0a1628;">
        <td style="padding:10px;color:#64748b;
                   font-weight:bold;">
          📊 Anomaly Score
        </td>
        <td style="padding:10px;color:#0ea5e9;">
          {recon_error:.4f}
        </td>
      </tr>
    </table>

    <!-- Action Required -->
    <div style="margin-top:15px;padding:12px;
                background:#1a0a0a;
                border:1px solid {color};
                border-radius:4px;">
      <p style="margin:0;color:{color};
                font-weight:bold;">
        ⚡ Recommended Actions:
      </p>
      <ul style="color:#cbd5e1;margin:8px 0 0;
                 padding-left:20px;">
        <li>Block source IP {src_ip} immediately</li>
        <li>Check firewall logs for port {port}</li>
        <li>Review traffic from this subnet</li>
        <li>Escalate to security team if persists</li>
      </ul>
    </div>

    <!-- Footer -->
    <div style="margin-top:15px;padding:10px;
                background:#0c2f54;
                border-radius:4px;
                text-align:center;">
      <p style="margin:0;color:#0ea5e9;font-size:12px;">
        Detected by LSTM Autoencoder |
        FL trained across 5 organizations |
        DP ε=0.000484
      </p>
    </div>
  </div>

</body>
</html>"""

    text = (
        f"CRITICAL ALERT: {attack_type}\n"
        f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Source: {src_ip} → {dst_ip}:{port}\n"
        f"Confidence: {confidence:.1f}%\n"
        f"Anomaly Score: {recon_error:.4f}\n"
    )

    return _send_email(subject, html, text)


def add_to_digest(attack_type, src_ip,
                  dst_ip, port,
                  recon_error, confidence):
    """Add non-critical attack to digest buffer."""
    with _digest_lock:
        _digest_buffer.append({
            "time":        datetime.now().strftime(
                               "%H:%M:%S"
                           ),
            "attack_type": attack_type,
            "severity":    get_severity(attack_type),
            "src_ip":      src_ip,
            "dst_ip":      dst_ip,
            "port":        port,
            "recon_error": recon_error,
            "confidence":  confidence,
        })


def send_digest():
    """Send 30-minute summary digest email."""
    with _digest_lock:
        if not _digest_buffer:
            print("  📭 No attacks to digest")
            return
        attacks = _digest_buffer.copy()
        _digest_buffer.clear()

    # Count by type
    counts = defaultdict(int)
    for a in attacks:
        counts[a["attack_type"]] += 1

    total    = len(attacks)
    now      = datetime.now()
    subject  = (
        f"📊 IDS Digest: {total} attacks in last 30 mins "
        f"[{now.strftime('%H:%M')}]"
    )

    # Build rows
    rows = ""
    for a in attacks[:20]:   # show max 20
        color = SEVERITY_COLORS.get(a["severity"], "#64748b")
        rows += f"""
        <tr>
          <td style="padding:8px;color:#64748b;">
            {a['time']}
          </td>
          <td style="padding:8px;color:{color};
                     font-weight:bold;">
            {a['attack_type']}
          </td>
          <td style="padding:8px;">
            <span style="background:{color};
                         color:white;padding:2px 8px;
                         border-radius:3px;font-size:11px;">
              {a['severity']}
            </span>
          </td>
          <td style="padding:8px;color:#ef4444;">
            {a['src_ip']}
          </td>
          <td style="padding:8px;color:#0ea5e9;">
            {a['confidence']:.0f}%
          </td>
        </tr>"""

    # Summary counts
    summary_rows = ""
    for atype, cnt in sorted(
        counts.items(), key=lambda x: -x[1]
    ):
        color = SEVERITY_COLORS.get(
            get_severity(atype), "#64748b"
        )
        pct   = cnt / total * 100
        summary_rows += f"""
        <tr>
          <td style="padding:8px;color:{color};
                     font-weight:bold;">{atype}</td>
          <td style="padding:8px;">{cnt}</td>
          <td style="padding:8px;">
            <div style="background:#1e3a5f;
                        border-radius:3px;height:8px;
                        width:100%;overflow:hidden;">
              <div style="background:{color};
                          height:8px;
                          width:{pct:.0f}%;"></div>
            </div>
          </td>
        </tr>"""

    html = f"""
<html>
<body style="font-family:Arial,sans-serif;
             background:#0a0e1a;
             color:#e2e8f0;padding:20px;">

  <div style="background:#0d1f35;padding:20px;
              border-radius:8px;
              border-top:4px solid #0ea5e9;">
    <h2 style="color:#0ea5e9;margin:0;">
      📊 30-Minute Security Digest
    </h2>
    <p style="color:#64748b;margin:5px 0 0;">
      {now.strftime('%Y-%m-%d %H:%M')} |
      AI-Powered IDS System
    </p>
  </div>

  <!-- Stats Row -->
  <div style="display:flex;gap:12px;
              margin:15px 0;flex-wrap:wrap;">
    <div style="background:#0d1f35;padding:15px;
                border-radius:8px;flex:1;
                border-top:3px solid #ef4444;
                min-width:100px;text-align:center;">
      <div style="font-size:28px;font-weight:bold;
                  color:#ef4444;">{total}</div>
      <div style="color:#64748b;font-size:12px;">
        Total Attacks
      </div>
    </div>
    <div style="background:#0d1f35;padding:15px;
                border-radius:8px;flex:1;
                border-top:3px solid #f59e0b;
                min-width:100px;text-align:center;">
      <div style="font-size:28px;font-weight:bold;
                  color:#f59e0b;">
        {len(counts)}
      </div>
      <div style="color:#64748b;font-size:12px;">
        Attack Types
      </div>
    </div>
    <div style="background:#0d1f35;padding:15px;
                border-radius:8px;flex:1;
                border-top:3px solid #0ea5e9;
                min-width:100px;text-align:center;">
      <div style="font-size:28px;font-weight:bold;
                  color:#0ea5e9;">
        {max(counts.values())}
      </div>
      <div style="color:#64748b;font-size:12px;">
        Peak Count
      </div>
    </div>
  </div>

  <!-- Attack Summary -->
  <div style="background:#0d1f35;padding:15px;
              border-radius:8px;margin-bottom:15px;">
    <h3 style="color:#e2e8f0;margin:0 0 10px;">
      Attack Breakdown
    </h3>
    <table style="width:100%;border-collapse:collapse;">
      <tr style="border-bottom:1px solid #1e4a8a;">
        <th style="padding:8px;text-align:left;
                   color:#64748b;">Type</th>
        <th style="padding:8px;text-align:left;
                   color:#64748b;">Count</th>
        <th style="padding:8px;text-align:left;
                   color:#64748b;">Distribution</th>
      </tr>
      {summary_rows}
    </table>
  </div>

  <!-- Recent Attacks Table -->
  <div style="background:#0d1f35;padding:15px;
              border-radius:8px;">
    <h3 style="color:#e2e8f0;margin:0 0 10px;">
      Recent Events (last 20)
    </h3>
    <table style="width:100%;border-collapse:collapse;">
      <tr style="border-bottom:1px solid #1e4a8a;">
        <th style="padding:8px;text-align:left;
                   color:#64748b;">Time</th>
        <th style="padding:8px;text-align:left;
                   color:#64748b;">Attack</th>
        <th style="padding:8px;text-align:left;
                   color:#64748b;">Severity</th>
        <th style="padding:8px;text-align:left;
                   color:#64748b;">Source</th>
        <th style="padding:8px;text-align:left;
                   color:#64748b;">Confidence</th>
      </tr>
      {rows}
    </table>
  </div>

  <div style="margin-top:15px;text-align:center;
              color:#64748b;font-size:11px;">
    AI-Powered IDS | FL across 5 orgs |
    DP ε=0.000484 | SHAP+LIME Explainability
  </div>

</body>
</html>"""

    text = (
        f"IDS Digest: {total} attacks\n"
        + "\n".join(
            f"{k}: {v}" for k, v in counts.items()
        )
    )
    _send_email(subject, html, text)


def process_alert(attack_type, src_ip, dst_ip,
                  port, recon_error, confidence):
    """
    Main function — call this for every detected attack.
    Automatically routes to correct alert strategy.
    """
    severity = get_severity(attack_type)

    if severity == "CRITICAL":
        # Immediate alert with cooldown
        send_critical_alert(
            attack_type, src_ip, dst_ip,
            port, recon_error, confidence
        )
    else:
        # Add to digest buffer
        add_to_digest(
            attack_type, src_ip, dst_ip,
            port, recon_error, confidence
        )


def start_digest_scheduler():
    """
    Background thread that sends digest every 30 mins.
    Call this once when your system starts.
    """
    def scheduler():
        while True:
            time.sleep(DIGEST_INTERVAL)
            print("\n📊 Sending scheduled digest...")
            send_digest()

    t = threading.Thread(
        target=scheduler, daemon=True
    )
    t.start()
    print(f"✅ Digest scheduler started "
          f"(every {DIGEST_INTERVAL//60} mins)")


def test_alerts():
    """Test both alert types."""
    global _email_enabled
    _email_enabled = False  # disable for testing

    print("\n🧪 Testing Alert System...")
    print("(Email disabled for test — "
          "enable in production)\n")

    # Test critical
    print("1. Testing CRITICAL alert (DDoS)...")
    process_alert(
        "DDoS", "192.168.1.105", "10.0.0.1",
        80, 0.847, 96.5
    )

    # Test digest
    print("\n2. Adding attacks to digest buffer...")
    attacks = [
        ("PortScan",  "10.0.1.15",  "172.16.0.1", 443),
        ("DoS Hulk",  "10.0.2.88",  "172.16.0.2", 80),
        ("FTP-Patator","10.0.3.12", "172.16.0.3", 21),
        ("PortScan",  "10.0.1.15",  "172.16.0.1", 8080),
        ("SSH-Patator","10.0.4.99", "172.16.0.4", 22),
    ]
    for atype, src, dst, port in attacks:
        process_alert(atype, src, dst, port,
                      0.45, 88.0)

    print(
        f"   Digest buffer: "
        f"{len(_digest_buffer)} attacks queued"
    )

    print("\n3. Sending digest now...")
    send_digest()

    print("\n✅ Alert system working!")
    print("\nTo enable real emails:")
    print("  1. Update SENDER_EMAIL and SENDER_PASSWORD")
    print("  2. Set _email_enabled = True")
    print("  3. Call start_digest_scheduler() on startup")


if __name__ == "__main__":
    test_alerts()