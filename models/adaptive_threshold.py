# ============================================================
# ADAPTIVE THRESHOLD — Smarter than fixed threshold
# Updates detection threshold based on recent traffic patterns
# This makes the IDS self-tuning and more robust
# ============================================================

import numpy as np
import torch
import pickle
import os
from collections import deque
from datetime import datetime

MODEL_DIR = r"C:\ids_project\models"
DATA_DIR  = r"C:\ids_project\data"

with open(f"{MODEL_DIR}/model_config.pkl", "rb") as f:
    config = pickle.load(f)

FIXED_THRESHOLD = config["threshold"]

class AdaptiveThreshold:
    """
    Automatically adjusts detection threshold
    based on recent traffic reconstruction errors.

    Why better than fixed threshold:
    - Network traffic patterns change over time
    - Morning traffic ≠ night traffic
    - Weekday traffic ≠ weekend traffic
    - Fixed threshold → too many false positives
      during unusual but legitimate traffic spikes
    - Adaptive threshold → adjusts to current baseline
    """

    def __init__(self,
                 window_size=1000,
                 sensitivity=2.5,
                 min_threshold=0.1,
                 max_threshold=0.9):
        """
        window_size  : number of recent packets to consider
        sensitivity  : how many std devs above mean = attack
                       (higher = less sensitive, fewer alerts)
                       (lower  = more sensitive, more alerts)
        min_threshold: never go below this
        max_threshold: never go above this
        """
        self.window      = deque(maxlen=window_size)
        self.sensitivity = sensitivity
        self.min_thresh  = min_threshold
        self.max_thresh  = max_threshold
        self.history     = []

        # Start with fixed threshold
        self.current = FIXED_THRESHOLD
        print(f"✅ Adaptive Threshold initialized")
        print(f"   Starting threshold : {FIXED_THRESHOLD:.4f}")
        print(f"   Window size        : {window_size}")
        print(f"   Sensitivity        : {sensitivity}σ")

    def update(self, recon_error):
        """
        Add new reconstruction error to window.
        Recalculate threshold based on recent normal traffic.
        """
        self.window.append(recon_error)

        if len(self.window) >= 50:
            errors = np.array(self.window)
            mean   = np.mean(errors)
            std    = np.std(errors)

            # New threshold = mean + N*std
            new_thresh = mean + self.sensitivity * std

            # Clip to valid range
            new_thresh = np.clip(
                new_thresh,
                self.min_thresh,
                self.max_thresh
            )

            self.current = float(new_thresh)

            self.history.append({
                "time":      datetime.now().strftime(
                                 "%H:%M:%S"
                             ),
                "threshold": self.current,
                "mean":      float(mean),
                "std":       float(std)
            })

        return self.current

    def is_attack(self, recon_error):
        """
        Classify packet using adaptive threshold.
        Also updates the threshold with this observation.
        """
        threshold  = self.update(recon_error)
        attack     = recon_error > threshold
        confidence = min(99.9, abs(
            recon_error - threshold
        ) / threshold * 100 + 50)

        return attack, threshold, confidence

    def get_stats(self):
        """Get current threshold statistics."""
        if len(self.window) == 0:
            return {}
        errors = np.array(self.window)
        return {
            "current_threshold": round(self.current, 4),
            "fixed_threshold":   round(FIXED_THRESHOLD, 4),
            "window_mean":       round(float(np.mean(errors)), 4),
            "window_std":        round(float(np.std(errors)), 4),
            "window_size":       len(self.window),
            "adjustment":        round(
                self.current - FIXED_THRESHOLD, 4
            )
        }

    def compare_with_fixed(self, X_seq, y_true,
                           model, device, benign_idx):
        """
        Compare adaptive vs fixed threshold performance.
        Shows why adaptive is better.
        """
        from sklearn.metrics import (
            accuracy_score, f1_score,
            precision_score, recall_score
        )

        print("\n📊 Comparing Fixed vs Adaptive Threshold...")

        errors = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(X_seq), 256):
                b = torch.FloatTensor(
                    X_seq[i:i+256]
                ).to(device)
                e = model.reconstruction_error(b)
                errors.extend(e.cpu().numpy())

        errors = np.array(errors)
        y_bin  = (y_true[:len(errors)] != benign_idx
                  ).astype(int)

        # Reset adaptive
        self.window.clear()
        self.current = FIXED_THRESHOLD

        # Fixed threshold
        fixed_preds  = (errors > FIXED_THRESHOLD).astype(int)
        fixed_f1     = f1_score(y_bin, fixed_preds,
                                zero_division=0)
        fixed_fp     = np.sum(
            (fixed_preds == 1) & (y_bin == 0)
        )

        # Adaptive threshold
        adaptive_preds = []
        for err in errors:
            is_atk, _, _ = self.is_attack(err)
            adaptive_preds.append(int(is_atk))

        adaptive_preds = np.array(adaptive_preds)
        adaptive_f1    = f1_score(y_bin, adaptive_preds,
                                  zero_division=0)
        adaptive_fp    = np.sum(
            (adaptive_preds == 1) & (y_bin == 0)
        )

        print(f"\n  Fixed Threshold ({FIXED_THRESHOLD:.4f}):")
        print(f"    F1-Score       : {fixed_f1*100:.2f}%")
        print(f"    False Positives: {fixed_fp:,}")

        print(f"\n  Adaptive Threshold:")
        print(f"    Final value    : {self.current:.4f}")
        print(f"    F1-Score       : {adaptive_f1*100:.2f}%")
        print(f"    False Positives: {adaptive_fp:,}")

        fp_reduction = fixed_fp - adaptive_fp
        print(f"\n  ✅ False Positive Reduction: "
              f"{fp_reduction:,} fewer false alarms!")

        return {
            "fixed_f1":       fixed_f1,
            "adaptive_f1":    adaptive_f1,
            "fixed_fp":       fixed_fp,
            "adaptive_fp":    adaptive_fp,
            "fp_reduction":   fp_reduction
        }


# ── Test adaptive threshold ──────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("  Adaptive Threshold — Self-Tuning IDS")
    print("="*60)

    at = AdaptiveThreshold(
        window_size  = 1000,
        sensitivity  = 2.5
    )

    # Simulate traffic
    print("\n🔄 Simulating live traffic...")
    print("   Phase 1: Normal business hours traffic")

    # Normal traffic — low reconstruction errors
    for _ in range(500):
        err = np.random.normal(0.15, 0.05)
        err = max(0.01, err)
        at.update(err)

    stats = at.get_stats()
    print(f"   Threshold after normal traffic: "
          f"{stats['current_threshold']}")

    print("\n   Phase 2: Attack traffic incoming...")

    # Attack traffic — high errors
    attack_errors = [
        np.random.uniform(0.6, 0.9)
        for _ in range(20)
    ]

    detected = 0
    for err in attack_errors:
        is_atk, thresh, conf = at.is_attack(err)
        if is_atk:
            detected += 1
            print(f"   🚨 ATTACK! Error={err:.3f} > "
                  f"Threshold={thresh:.4f} "
                  f"({conf:.1f}% confidence)")

    print(f"\n   Detected {detected}/{len(attack_errors)} "
          f"attacks")

    print("\n   Phase 3: Night traffic (different pattern)")
    for _ in range(300):
        err = np.random.normal(0.08, 0.02)
        err = max(0.01, err)
        at.update(err)

    stats = at.get_stats()
    print(f"   Threshold adjusted for night: "
          f"{stats['current_threshold']}")
    print(f"   (Fixed threshold was: {FIXED_THRESHOLD:.4f})")
    print(f"   Adjustment: {stats['adjustment']:+.4f}")

    print(f"\n{'='*60}")
    print(f"✅ ADAPTIVE THRESHOLD COMPLETE!")
    print(f"   The threshold self-adjusts to traffic patterns")
    print(f"   Fewer false alarms during unusual but")
    print(f"   legitimate traffic spikes")
    print(f"{'='*60}")

    # Save for use in dashboard
    with open(
        f"{MODEL_DIR}/adaptive_threshold.pkl", "wb"
    ) as f:
        pickle.dump(at, f)
    print(f"\n✅ Saved to: {MODEL_DIR}/adaptive_threshold.pkl")