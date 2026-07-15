# ============================================================
# SQLITE ATTACK LOGGER
# Logs every detected attack to a local database
# Enables forensic analysis, reporting, and history queries
# ============================================================

import sqlite3
import os
from datetime import datetime
import pandas as pd

DB_PATH = r"C:\ids_project\database\attacks.db"
os.makedirs(r"C:\ids_project\database", exist_ok=True)

def init_db():
    """Create database and tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS attacks (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            attack_type   TEXT NOT NULL,
            severity      TEXT NOT NULL,
            src_ip        TEXT,
            dst_ip        TEXT,
            port          INTEGER,
            recon_error   REAL,
            confidence    REAL,
            fl_round      INTEGER DEFAULT 0,
            alerted       INTEGER DEFAULT 0
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS fl_rounds (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            round_num     INTEGER,
            accuracy      REAL,
            f1_score      REAL,
            dp_epsilon    REAL,
            clients       INTEGER
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS system_stats (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            total_packets   INTEGER,
            total_attacks   INTEGER,
            attack_rate     REAL,
            model_threshold REAL
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Database initialized at:", DB_PATH)


def log_attack(attack_type, src_ip, dst_ip,
               port, recon_error, confidence,
               fl_round=0):
    """Log a detected attack to the database."""

    # Determine severity
    if attack_type == "BENIGN":
        return  # Don't log normal traffic

    if attack_type in ["DDoS", "Bot", "Infiltration"]:
        severity = "CRITICAL"
    elif attack_type in ["DoS Hulk", "DoS GoldenEye",
                          "DoS slowloris", "DoS Slowhttptest"]:
        severity = "HIGH"
    elif attack_type in ["PortScan", "FTP-Patator",
                          "SSH-Patator"]:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute("""
        INSERT INTO attacks
        (timestamp, attack_type, severity, src_ip,
         dst_ip, port, recon_error, confidence, fl_round)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        attack_type, severity, src_ip, dst_ip,
        port, recon_error, confidence, fl_round
    ))

    conn.commit()
    conn.close()


def log_fl_round(round_num, accuracy,
                 f1_score, dp_epsilon, clients=5):
    """Log FL training round results."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute("""
        INSERT INTO fl_rounds
        (timestamp, round_num, accuracy,
         f1_score, dp_epsilon, clients)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        round_num, accuracy, f1_score,
        dp_epsilon, clients
    ))

    conn.commit()
    conn.close()


def get_attack_summary(hours=24):
    """Get attack summary for last N hours."""
    conn = sqlite3.connect(DB_PATH)

    query = f"""
        SELECT
            attack_type,
            severity,
            COUNT(*) as count,
            AVG(confidence) as avg_confidence,
            MAX(recon_error) as max_error,
            MIN(timestamp) as first_seen,
            MAX(timestamp) as last_seen
        FROM attacks
        WHERE timestamp >= datetime(
            'now', '-{hours} hours'
        )
        GROUP BY attack_type, severity
        ORDER BY count DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_recent_attacks(limit=50):
    """Get most recent attacks."""
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(f"""
        SELECT * FROM attacks
        ORDER BY timestamp DESC
        LIMIT {limit}
    """, conn)
    conn.close()
    return df


def get_attack_timeline(hours=24):
    """Get attack counts per hour for timeline chart."""
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(f"""
        SELECT
            strftime('%H:00', timestamp) as hour,
            attack_type,
            COUNT(*) as count
        FROM attacks
        WHERE timestamp >= datetime(
            'now', '-{hours} hours'
        )
        GROUP BY hour, attack_type
        ORDER BY hour
    """, conn)
    conn.close()
    return df


def get_top_sources(limit=10):
    """Get top attacking IP addresses."""
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(f"""
        SELECT
            src_ip,
            COUNT(*) as attack_count,
            GROUP_CONCAT(
                DISTINCT attack_type
            ) as attack_types,
            MAX(timestamp) as last_seen
        FROM attacks
        WHERE attack_type != 'BENIGN'
        GROUP BY src_ip
        ORDER BY attack_count DESC
        LIMIT {limit}
    """, conn)
    conn.close()
    return df


def export_to_csv(filepath=None):
    """Export all attacks to CSV for reporting."""
    if filepath is None:
        filepath = (
            f"C:\\ids_project\\database\\"
            f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f".csv"
        )
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(
        "SELECT * FROM attacks ORDER BY timestamp DESC",
        conn
    )
    conn.close()
    df.to_csv(filepath, index=False)
    print(f"✅ Exported {len(df)} records to: {filepath}")
    return filepath


def get_stats():
    """Get overall database statistics."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute("SELECT COUNT(*) FROM attacks")
    total = c.fetchone()[0]

    c.execute("""
        SELECT COUNT(*) FROM attacks
        WHERE timestamp >= datetime('now', '-1 hour')
    """)
    last_hour = c.fetchone()[0]

    c.execute("""
        SELECT attack_type, COUNT(*) as cnt
        FROM attacks
        GROUP BY attack_type
        ORDER BY cnt DESC
        LIMIT 1
    """)
    top = c.fetchone()

    c.execute("""
        SELECT COUNT(DISTINCT src_ip) FROM attacks
    """)
    unique_ips = c.fetchone()[0]

    conn.close()
    return {
        "total_attacks":  total,
        "last_hour":      last_hour,
        "top_attack":     top[0] if top else "None",
        "unique_sources": unique_ips
    }


# Initialize on import
init_db()

if __name__ == "__main__":
    print("\n🗄️ Testing Attack Logger...")

    # Insert sample data
    import random
    attack_types = [
        "DDoS", "PortScan", "Bot",
        "DoS Hulk", "Infiltration"
    ]

    for i in range(20):
        attack = random.choice(attack_types)
        log_attack(
            attack_type  = attack,
            src_ip       = f"192.168.{random.randint(1,10)}"
                           f".{random.randint(1,254)}",
            dst_ip       = f"10.0.0.{random.randint(1,10)}",
            port         = random.choice(
                               [80, 443, 22, 8080]
                           ),
            recon_error  = random.uniform(0.4, 0.9),
            confidence   = random.uniform(85, 99),
        )

    print("\n📊 Attack Summary:")
    print(get_attack_summary())

    print("\n🔝 Top Sources:")
    print(get_top_sources())

    print("\n📈 Stats:")
    stats = get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    export_to_csv()
    print("\n✅ Logger working perfectly!")