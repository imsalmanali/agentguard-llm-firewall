"""
AgentGuard AI - Model Training, Benchmarking & Artifact Generation Pipeline.
Generates benchmark datasets, trains the multi-tier injection classifier,
computes evaluation metrics, produces diagnostic plots, and serializes artifacts.
"""

import os
import sys

# Ensure repository root is in python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

from src.data_generator import generate_security_benchmark_dataset
from src.detector import PromptInjectionDetector


def run_training_pipeline():
    print("=" * 70)
    print("[AgentGuard AI] Initiating Production Training & Benchmark Pipeline")
    print("=" * 70)

    # 1. Dataset Generation
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    artifacts_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    print("[*] Generating 3,000 synthetic adversarial & benign security prompts...")
    df = generate_security_benchmark_dataset(n_samples=3000, random_state=42)
    csv_path = os.path.join(data_dir, "adversarial_prompt_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] Benchmark dataset saved to {csv_path} ({len(df)} samples)")
    print("   Class Breakdown:")
    for cat, cnt in df["category"].value_counts().items():
        print(f"   - {cat:<26}: {cnt} samples")

    # 2. Train/Test Split
    X = np.array(df["text"].tolist(), dtype=object)
    y = np.array(df["is_threat"].tolist(), dtype=int)
    categories = np.array(df["category"].tolist(), dtype=object)

    X_train, X_test, y_train, y_test, cat_train, cat_test = train_test_split(
        X, y, categories, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[*] Data partitioned: {len(X_train)} training samples, {len(X_test)} validation samples.")

    # 3. Model Training
    detector = PromptInjectionDetector()
    print("[*] Fitting TF-IDF N-gram feature pipeline and Cost-Sensitive Logistic Regressor...")
    t0 = time.time()
    detector.fit(X_train.tolist(), y_train.tolist())
    train_duration = time.time() - t0
    print(f"[OK] Training completed in {train_duration:.2f} seconds.")

    # 4. Evaluation & Metrics
    print("[*] Evaluating model performance on held-out validation set...")
    y_pred_probs = []
    y_preds = []
    latencies = []

    for text in X_test:
        t_start = time.perf_counter()
        res = detector.predict(text)
        lat_ms = (time.perf_counter() - t_start) * 1000.0
        latencies.append(lat_ms)
        y_pred_probs.append(res["threat_probability"])
        y_preds.append(1 if res["is_threat"] else 0)

    y_pred_probs = np.array(y_pred_probs)
    y_preds = np.array(y_preds)

    acc = float(accuracy_score(y_test, y_preds))
    prec = float(precision_score(y_test, y_preds, zero_division=0))
    rec = float(recall_score(y_test, y_preds, zero_division=0))
    f1 = float(f1_score(y_test, y_preds, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_pred_probs))
    pr_auc = float(average_precision_score(y_test, y_pred_probs))
    cm = confusion_matrix(y_test, y_preds).tolist()

    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))
    avg_lat = float(np.mean(latencies))

    print("\n[+] Benchmark Results:")
    print(f"   * Accuracy:               {acc * 100:.2f}%")
    print(f"   * Precision:              {prec * 100:.2f}%")
    print(f"   * Recall:                 {rec * 100:.2f}%")
    print(f"   * F1-Score:               {f1 * 100:.2f}%")
    print(f"   * ROC-AUC:                {roc_auc:.4f}")
    print(f"   * PR-AUC (Avg Precision): {pr_auc:.4f}")
    print(f"   * Latency:                Avg={avg_lat:.2f}ms | p50={p50_lat:.2f}ms | p95={p95_lat:.2f}ms | p99={p99_lat:.2f}ms")

    # 5. Serialize Artifacts
    model_save_path = os.path.join(artifacts_dir, "injection_detector.joblib")
    detector.save(model_save_path)
    print(f"\n[OK] Model serialized to: {model_save_path}")

    metrics_dict = {
        "model_name": "AgentGuard-PromptInjection-v1",
        "dataset_samples": len(df),
        "test_samples": len(X_test),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "confusion_matrix": cm
        },
        "latency_benchmarks_ms": {
            "average": round(avg_lat, 3),
            "median_p50": round(p50_lat, 3),
            "percentile_p95": round(p95_lat, 3),
            "percentile_p99": round(p99_lat, 3)
        },
        "category_distribution": df["category"].value_counts().to_dict()
    }

    metrics_save_path = os.path.join(artifacts_dir, "benchmark_metrics.json")
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"[OK] Metrics saved to: {metrics_save_path}")

    # 6. Generate Diagnostic Visualizations
    print("\n[*] Rendering diagnostic plots and figures...")
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Subplot 1: Confusion Matrix
    cm_arr = np.array(cm)
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0, 0],
        xticklabels=["Predicted Benign", "Predicted Threat"],
        yticklabels=["Actual Benign", "Actual Threat"]
    )
    axes[0, 0].set_title(f"Confusion Matrix (Acc: {acc*100:.1f}%)", fontsize=12, fontweight="bold")

    # Subplot 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    axes[0, 1].plot(fpr, tpr, color="#009688", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    axes[0, 1].plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel("False Positive Rate (FPR)")
    axes[0, 1].set_ylabel("True Positive Rate (TPR / Recall)")
    axes[0, 1].set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    axes[0, 1].legend(loc="lower right")

    # Subplot 3: Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_probs)
    axes[1, 0].plot(recall_vals, precision_vals, color="#E91E63", lw=2.5, label=f"PR Curve (PR-AUC = {pr_auc:.4f})")
    axes[1, 0].set_xlabel("Recall")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    axes[1, 0].legend(loc="lower left")

    # Subplot 4: Threat Category Breakdown
    cat_series = df["category"].value_counts()
    axes[1, 1].barh(cat_series.index, cat_series.values, color=["#4CAF50", "#F44336", "#FF9800", "#9C27B0", "#2196F3", "#00BCD4"])
    axes[1, 1].set_xlabel("Sample Count")
    axes[1, 1].set_title("Threat & Benign Benchmark Corpus Distribution", fontsize=12, fontweight="bold")

    plt.tight_layout()
    diag_plot_path = os.path.join(artifacts_dir, "model_diagnostics.png")
    fig.savefig(diag_plot_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Model diagnostics saved to: {diag_plot_path}")

    # Latency Distribution Plot
    fig_lat, ax_lat = plt.subplots(figsize=(9, 5))
    sns.histplot(latencies, bins=30, kde=True, color="#3F51B5", ax=ax_lat)
    ax_lat.axvline(avg_lat, color="red", linestyle="--", label=f"Mean Latency ({avg_lat:.2f}ms)")
    ax_lat.axvline(p95_lat, color="orange", linestyle=":", label=f"p95 Latency ({p95_lat:.2f}ms)")
    ax_lat.set_xlabel("Inference Latency per Query (milliseconds)")
    ax_lat.set_ylabel("Frequency")
    ax_lat.set_title("AgentGuard Gateway Latency Distribution Overhead", fontsize=12, fontweight="bold")
    ax_lat.legend()
    plt.tight_layout()
    lat_plot_path = os.path.join(artifacts_dir, "latency_overhead_distribution.png")
    fig_lat.savefig(lat_plot_path, dpi=200)
    plt.close(fig_lat)
    print(f"[OK] Latency distribution saved to: {lat_plot_path}")

    print("\n[SUCCESS] Training and Artifact Generation successfully finalized!")
    return metrics_dict


if __name__ == "__main__":
    run_training_pipeline()
