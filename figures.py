"""Generate deterministic figures from repaired P1 CSV outputs."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join("results", ".mplconfig"))

import matplotlib.pyplot as plt

from config import RESULTS_DIR, WINDOW_MS
from results import read_csv


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def plot_e1(results_dir: str = RESULTS_DIR) -> None:
    rows = read_csv(os.path.join(results_dir, "e1_storage.csv"))
    n = [_float(row, "n") for row in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(n, [_float(row, "aes_bytes") for row in rows], marker="o", label="AES per reading")
    plt.plot(n, [_float(row, "paillier_nobatch_bytes") for row in rows], marker="^", label="Paillier no-batch")
    plt.plot(n, [_float(row, "ours_paillier_bytes") for row in rows], marker="D", label="Ours Paillier aggregate")
    plt.yscale("log")
    plt.xlabel("Sensors per window")
    plt.ylabel("Bytes per window")
    plt.title("E1 Storage: byte reduction reported separately from ciphertext count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "e1_storage.png"))
    plt.savefig(os.path.join(results_dir, "e1_storage.pdf"))
    plt.close()


def plot_e2(results_dir: str = RESULTS_DIR) -> None:
    rows = read_csv(os.path.join(results_dir, "e2_latency.csv"))
    n = [_float(row, "n") for row in rows]
    plt.figure(figsize=(7, 4))
    for key, label in [
        ("plaintext_ms_median", "Plaintext"),
        ("aes_ms_median", "AES decrypt+sum"),
        ("paillier_nobatch_ms_median", "Paillier no-batch"),
        ("ours_ms_median", "Ours"),
    ]:
        plt.plot(n, [_float(row, key) for row in rows], marker="o", label=label)
    plt.axhline(WINDOW_MS, color="red", linestyle="--", label="500 ms window")
    plt.xlabel("Sensors per window")
    plt.ylabel("Median latency (ms)")
    plt.title("E2 Latency: measured feasibility vs 500 ms window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "e2_latency.png"))
    plt.savefig(os.path.join(results_dir, "e2_latency.pdf"))
    plt.close()


def plot_e3a(results_dir: str = RESULTS_DIR) -> None:
    rows = read_csv(os.path.join(results_dir, "e3a_correctness.csv"))
    k = [str(int(_float(row, "k_delegated"))) for row in rows]
    plt.figure(figsize=(7, 4))
    plt.bar(k, [_float(row, "accuracy_scaled_pct") for row in rows], color="#1D9E75")
    plt.ylim(95, 101)
    plt.xlabel("Delegated sensors")
    plt.ylabel("Scaled-sum correctness (%)")
    plt.title("E3a Correctness: n=100, 100 trials per k")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "e3a_correctness.png"))
    plt.savefig(os.path.join(results_dir, "e3a_correctness.pdf"))
    plt.close()


def plot_e5(results_dir: str = RESULTS_DIR) -> None:
    rows = read_csv(os.path.join(results_dir, "e5_capacity_score.csv"))
    labels = [row["method"] for row in rows]
    x = range(len(labels))
    plt.figure(figsize=(8, 4))
    plt.bar(x, [_float(row, "deadline_satisfaction_pct_mean") for row in rows], color="#378ADD")
    plt.xticks(list(x), labels, rotation=15, ha="right")
    plt.ylabel("Deadline satisfaction mean (%)")
    plt.title("E5 Task-Level Evaluation Across Fixed Seeds")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "e5_capacity_score.png"))
    plt.savefig(os.path.join(results_dir, "e5_capacity_score.pdf"))
    plt.close()


def generate_all(results_dir: str = RESULTS_DIR) -> None:
    plot_e1(results_dir)
    plot_e2(results_dir)
    plot_e3a(results_dir)
    plot_e5(results_dir)


if __name__ == "__main__":
    generate_all()
