"""Generate deterministic figures from repaired P1 CSV outputs."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join("results", ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np

from config import FOG_NODES, KMM_PROV_MS, RESULTS_DIR, WINDOW_MS
from results import read_csv

C = {
    "yours": "#1D9E75",
    "aes": "#378ADD",
    "paillier": "#D85A30",
    "plain": "#888780",
    "rr": "#7F77DD",
    "thresh": "#BA7517",
    "random": "#888780",
}

METHOD_LABELS = {
    "random": "Random",
    "round_robin": "Round-Robin",
    "threshold": "Threshold only",
    "capacity": "CapacityScore (ours)",
}

METHOD_COLORS = {
    "random": C["random"],
    "round_robin": C["rr"],
    "threshold": C["thresh"],
    "capacity": C["yours"],
}


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _finish_figure(results_dir: str, basename: str, show: bool) -> None:
    plt.savefig(os.path.join(results_dir, f"{basename}.png"))
    plt.savefig(os.path.join(results_dir, f"{basename}.pdf"))
    if show:
        plt.show()
    else:
        plt.close()


def plot_e1(results_dir: str = RESULTS_DIR, show: bool = False) -> None:
    rows = read_csv(os.path.join(results_dir, "e1_storage.csv"))
    n = [_float(row, "n") for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(n, [_float(row, "plaintext_bytes") for row in rows], color=C["plain"], marker="s", lw=1.5, label="Plaintext (insecure)")
    ax.plot(n, [_float(row, "aes_bytes") for row in rows], color=C["aes"], marker="o", lw=1.5, label="AES per reading")
    ax.plot(n, [_float(row, "paillier_nobatch_bytes") for row in rows], color=C["paillier"], marker="^", lw=1.5, label="Paillier no-batch")
    ax.plot(n, [_float(row, "ours_paillier_bytes") for row in rows], color=C["yours"], marker="D", lw=2.5, label="Ours: 1 Paillier aggregate")
    ax.set_xlabel("Number of sensors per fog node (n)")
    ax.set_ylabel("Storage per window (bytes, log scale)")
    ax.set_title("E1a - Storage per window vs n")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(n, [_float(row, "aes_ciphertexts") for row in rows], color=C["aes"], marker="o", lw=1.5, label="AES per reading")
    ax2.plot(n, [_float(row, "paillier_nobatch_ciphertexts") for row in rows], color=C["paillier"], marker="^", lw=1.5, label="Paillier no-batch")
    ax2.plot(n, [_float(row, "ours_ciphertexts") for row in rows], color=C["yours"], marker="D", lw=2.5, label="Ours")
    ax2.set_xlabel("Number of sensors per fog node (n)")
    ax2.set_ylabel("Ciphertexts transmitted per window")
    ax2.set_title("E1b - Ciphertext count vs n")
    ax2.legend(fontsize=8)

    fig.suptitle("E1 - Storage Reduction: Paillier Slot Aggregation", fontweight="bold", y=1.01)
    plt.tight_layout()
    _finish_figure(results_dir, "e1_storage", show)


def plot_e2(results_dir: str = RESULTS_DIR, show: bool = False) -> None:
    rows = read_csv(os.path.join(results_dir, "e2_latency.csv"))
    n = [_float(row, "n") for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    for key, label, color, lw in [
        ("plaintext_ms_median", "Plaintext sum", C["plain"], 1.5),
        ("aes_ms_median", "AES decrypt+sum", C["aes"], 1.5),
        ("paillier_nobatch_ms_median", "Paillier no-batch", C["paillier"], 1.5),
        ("ours_ms_median", "Ours", C["yours"], 2.5),
    ]:
        yerr_key = key.replace("_median", "_std")
        yerr = [_float(row, yerr_key) for row in rows] if yerr_key in rows[0] else None
        ax.errorbar(n, [_float(row, key) for row in rows], yerr=yerr, marker="o", color=color, lw=lw, capsize=3, label=label)
    ax.axhline(WINDOW_MS, color="red", linestyle="--", lw=1, alpha=0.7, label="500 ms window")
    ax.set_xlabel("Sensors per fog node (n)")
    ax.set_ylabel("Median window latency (ms)")
    ax.set_title("E2a - Latency vs n")
    ax.legend(fontsize=7.5)

    reference = next((row for row in rows if int(_float(row, "n")) == 100), rows[-1])
    reference_n = int(_float(reference, "n"))
    base_latency = _float(reference, "ours_ms_median")
    node_ids = ["F1", "F3", "F4"]
    node_meds = [base_latency * FOG_NODES[node_id]["speed_factor"] for node_id in node_ids]
    ax2 = axes[1]
    bars = ax2.bar(node_ids, node_meds, color=[C["yours"], C["paillier"], C["aes"]], width=0.45, edgecolor="white")
    ax2.axhline(WINDOW_MS, color="red", linestyle="--", lw=1, alpha=0.7, label="500 ms window")
    for bar, med in zip(bars, node_meds):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{med:.0f} ms", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(range(len(node_ids)))
    ax2.set_xticklabels([f"{node_id}\n({FOG_NODES[node_id]['class']})" for node_id in node_ids])
    ax2.set_ylabel("Estimated latency (ms)")
    ax2.set_title(f"E2b - Node heterogeneity (n={reference_n})")
    ax2.legend(fontsize=8)

    ax3 = axes[2]
    enc = _float(reference, "ours_enclave_ms_median")
    kmm = _float(reference, "ours_kmm_ms_median")
    storage = _float(reference, "ours_storage_ms_median")
    labels = ["Ours\n(normal)", "Ours\n(+delegation)"]
    x = np.arange(len(labels))
    width = 0.4
    ax3.bar(x, [enc, enc], width, label="Enclave AES->Paillier", color=C["yours"], edgecolor="white")
    ax3.bar(x, [kmm, kmm], width, bottom=[enc, enc], label="KMM combine", color=C["aes"], edgecolor="white")
    ax3.bar(x, [storage, storage], width, bottom=[enc + kmm, enc + kmm], label="Storage prep", color=C["rr"], edgecolor="white")
    ax3.bar(x, [0, KMM_PROV_MS], width, bottom=[enc + kmm + storage, enc + kmm + storage], label="KMM provisioning", color=C["paillier"], edgecolor="white")
    ax3.axhline(WINDOW_MS, color="red", linestyle="--", lw=1, alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel("Latency (ms)")
    ax3.set_title(f"E2c - Latency breakdown (n={reference_n})")
    ax3.legend(fontsize=7)

    fig.suptitle("E2 - Aggregation Latency: Full Pipeline vs Baselines", fontweight="bold", y=1.01)
    plt.tight_layout()
    _finish_figure(results_dir, "e2_latency", show)


def plot_e3a(results_dir: str = RESULTS_DIR, show: bool = False) -> None:
    rows = read_csv(os.path.join(results_dir, "e3a_correctness.csv"))
    k = [str(int(_float(row, "k_delegated"))) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    bars = ax.bar(k, [_float(row, "accuracy_scaled_pct") for row in rows], color=C["yours"], width=0.5, edgecolor="white")
    ax.axhline(100, color="green", linestyle="--", lw=1.5, alpha=0.7, label="100% target")
    ax.set_ylim(95, 101.5)
    ax.set_xlabel("k delegated")
    ax.set_ylabel("Scaled-sum correctness (%)")
    ax.set_title("E3a - Slot protocol correctness")
    for bar, row in zip(bars, rows):
        val = _float(row, "accuracy_scaled_pct")
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05, f"{val:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.axis("off")
    table_data = [
        [
            str(int(_float(row, "k_delegated"))),
            f"{int(_float(row, 'correct_scaled'))}/{int(_float(row, 'trials'))}",
            f"{_float(row, 'accuracy_scaled_pct'):.0f}%",
            f"{_float(row, 'median_quantization_error'):.4f}",
            f"{_float(row, 'median_trial_ms'):.0f} ms",
        ]
        for row in rows
    ]
    columns = ["k", "Correct", "Accuracy", "Median quant. err", "Trial latency"]
    table = ax2.table(cellText=table_data, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.15, 1.7)
    for (row_idx, _), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor(C["yours"])
            cell.set_text_props(color="white", fontweight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f0faf6")
    ax2.set_title("E3a - Results summary\nzero-fill security argued, not empirically verified", pad=12)

    fig.suptitle("E3a - Slot Protocol Correctness Under Mid-Window Delegation", fontweight="bold", y=1.01)
    plt.tight_layout()
    _finish_figure(results_dir, "e3a_correctness", show)


def plot_e5(results_dir: str = RESULTS_DIR, show: bool = False) -> None:
    rows = read_csv(os.path.join(results_dir, "e5_capacity_score.csv"))
    labels = [METHOD_LABELS[row["method"]] for row in rows]
    bar_colors = [METHOD_COLORS[row["method"]] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def annotate(ax, values: list[float], lower_better: bool = False) -> None:
        best = int(np.argmin(values)) if lower_better else int(np.argmax(values))
        for i, (bar, val) in enumerate(zip(ax.patches, values)):
            weight = "bold" if i == best else "normal"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01, f"{val:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight=weight)

    ax = axes[0, 0]
    values = [_float(row, "workload_stdev_mean") for row in rows]
    ax.bar(labels, values, color=bar_colors, width=0.5, edgecolor="white")
    annotate(ax, values, lower_better=True)
    ax.set_title("Workload std deviation\n(lower = more balanced)")
    ax.set_ylabel("Std dev")
    ax.tick_params(axis="x", labelrotation=12)

    ax = axes[0, 1]
    values = [_float(row, "completion_pct_mean") for row in rows]
    ax.bar(labels, values, color=bar_colors, width=0.5, edgecolor="white")
    annotate(ax, values)
    ax.set_title("Task completion rate\n(higher = better)")
    ax.set_ylabel("Completion (%)")
    ax.tick_params(axis="x", labelrotation=12)

    ax = axes[1, 0]
    values = [_float(row, "redelegation_rate_pct_mean") for row in rows]
    ax.bar(labels, values, color=bar_colors, width=0.5, edgecolor="white")
    annotate(ax, values, lower_better=True)
    ax.set_title("Re-delegation rate\n(lower = better)")
    ax.set_ylabel("Rate (%)")
    ax.tick_params(axis="x", labelrotation=12)

    ax = axes[1, 1]
    x = np.arange(len(rows))
    width = 0.35
    ax.bar(x - width / 2, [_float(row, "ls_target_accuracy_pct_mean") for row in rows], width, label="LS -> F4", color=bar_colors, edgecolor="white")
    ax.bar(x + width / 2, [_float(row, "to_target_accuracy_pct_mean") for row in rows], width, label="TO -> F1", color=bar_colors, alpha=0.45, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Target accuracy (%)")
    ax.set_title("Task-aware target accuracy")
    ax.legend(fontsize=8)

    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(True, alpha=0.3)

    fig.suptitle("E5 - CapacityScore vs Delegation Baselines\nTask-level simulation across fixed seeds", fontweight="bold", y=1.02)
    plt.tight_layout()
    _finish_figure(results_dir, "e5_capacity_score", show)


def generate_all(results_dir: str = RESULTS_DIR, show: bool = False) -> None:
    plot_e1(results_dir, show=show)
    plot_e2(results_dir, show=show)
    plot_e3a(results_dir, show=show)
    plot_e5(results_dir, show=show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate P1 figures from CSV results")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--show", action="store_true", help="Display figures with matplotlib after saving")
    args = parser.parse_args()
    generate_all(args.results_dir, show=args.show)
