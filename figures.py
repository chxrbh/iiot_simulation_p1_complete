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
    "cloud_only": "Cloud-only",
    "fog_plaintext": "Fog plaintext",
    "paillier_fog_convert": "Paillier\nfog convert",
    "ours": "Proposed",
}

METHOD_COLORS = {
    "random": C["random"],
    "round_robin": C["rr"],
    "threshold": C["thresh"],
    "capacity": C["yours"],
}

E6_METHOD_LABELS = {
    "b1_gossip": "Gossip-only",
    "b2_replication": "Replication",
    "checkpoint": "Checkpoint",
    "b4_multilayer": "Multilayer",
    "b5_fog_clustering": "Fog-clustering",
    "proposed_ack_kmm": "Proposed\nACK+KMM",
}

E6_METHOD_COLORS = {
    "b1_gossip": C["plain"],
    "b2_replication": C["paillier"],
    "checkpoint": C["thresh"],
    "b4_multilayer": C["aes"],
    "b5_fog_clustering": C["rr"],
    "proposed_ack_kmm": C["yours"],
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
    ax.bar(x - width / 2, [_float(row, "ls_policy_agreement_pct_mean") for row in rows], width, label="LS policy -> F4", color=bar_colors, edgecolor="white")
    ax.bar(x + width / 2, [_float(row, "to_policy_agreement_pct_mean") for row in rows], width, label="TO policy -> F1", color=bar_colors, alpha=0.45, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Synthetic policy agreement (%)")
    ax.set_title("Task-policy agreement\n(not ground-truth accuracy)")
    ax.legend(fontsize=8)

    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(True, alpha=0.3)

    fig.suptitle("E5 - CapacityScore vs Heuristic Delegation Baselines\nTask-level simulation across fixed seeds", fontweight="bold", y=1.02)
    plt.tight_layout()
    _finish_figure(results_dir, "e5_capacity_score", show)


def plot_e5_sensitivity(results_dir: str = RESULTS_DIR, show: bool = False) -> None:
    rows = read_csv(os.path.join(results_dir, "e5_capacity_score_sensitivity.csv"))
    labels = [row["weight_variant"].replace("_", "\n") for row in rows]
    x = np.arange(len(rows))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(x - width / 2, [_float(row, "deadline_satisfaction_pct_mean") for row in rows], width, label="Deadline met", color=C["yours"], edgecolor="white")
    axes[0].bar(x + width / 2, [_float(row, "completion_pct_mean") for row in rows], width, label="Completion", color=C["aes"], edgecolor="white")
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_title("CapacityScore weight sensitivity")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend(fontsize=8)

    axes[1].bar(x, [_float(row, "workload_stdev_mean") for row in rows], color=C["paillier"], width=0.45, edgecolor="white")
    axes[1].set_ylabel("Mean workload std dev")
    axes[1].set_title("Balance under alternate weights")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(True, alpha=0.3)
    plt.tight_layout()
    _finish_figure(results_dir, "e5_capacity_score_sensitivity", show)


def plot_e3b(results_dir: str = RESULTS_DIR, show: bool = False) -> None:
    rows = read_csv(os.path.join(results_dir, "e3b_multisource_correctness.csv"))
    row = rows[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    accuracy = _float(row, "accuracy_scaled_pct")
    ax.bar(["F1+F2 -> F4"], [accuracy], color=C["yours"], width=0.45, edgecolor="white")
    ax.axhline(100, color="green", linestyle="--", lw=1.2, alpha=0.7)
    ax.set_ylim(95, 101.5)
    ax.set_ylabel("Scaled-sum correctness (%)")
    ax.set_title("E3b - Multi-source arithmetic correctness")
    ax.text(0, accuracy + 0.05, f"{accuracy:.0f}%", ha="center", va="bottom", fontweight="bold")

    ax2 = axes[1]
    ax2.axis("off")
    table_data = [
        ["Total sensors", str(int(_float(row, "total_sensors")))],
        ["Trials", str(int(_float(row, "trials")))],
        ["Correct", f"{int(_float(row, 'correct_scaled'))}/{int(_float(row, 'trials'))}"],
        ["Provisioned", row["provisioned_edges"]],
        ["Median quant. err", f"{_float(row, 'median_quantization_error'):.4f}"],
    ]
    table = ax2.table(cellText=table_data, colLabels=["Metric", "Value"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.6)
    ax2.set_title("E3b summary\nnot a security proof", pad=12)
    plt.tight_layout()
    _finish_figure(results_dir, "e3b_multisource_correctness", show)


def plot_e4(results_dir: str = RESULTS_DIR, show: bool = False) -> None:
    rows = read_csv(os.path.join(results_dir, "e4_kmm_combine.csv"))
    k_values = [_float(row, "k_fog_aggregates") for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.errorbar(
        k_values,
        [_float(row, "combine_latency_ms_median") for row in rows],
        yerr=[_float(row, "combine_latency_ms_std") for row in rows],
        marker="o",
        color=C["yours"],
        capsize=3,
    )
    ax.axhline(WINDOW_MS, color="red", linestyle="--", lw=1, alpha=0.7, label="500 ms window")
    ax.set_xlabel("Fog aggregates combined (k)")
    ax.set_ylabel("KMM combine latency (ms)")
    ax.set_title("E4a - KMM combine latency\nHE additions only")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(k_values, [_float(row, "bytes_received") for row in rows], marker="s", color=C["paillier"])
    ax2.set_xlabel("Fog aggregates combined (k)")
    ax2.set_ylabel("Bytes received by KMM")
    ax2.set_title("E4b - KMM input size")
    plt.tight_layout()
    _finish_figure(results_dir, "e4_kmm_combine", show)


def plot_e6(results_dir: str = RESULTS_DIR, show: bool = False) -> None:
    rows = read_csv(os.path.join(results_dir, "e6_fault_detection.csv"))
    scenarios = list(dict.fromkeys(row["scenario"] for row in rows))
    methods = list(dict.fromkeys(row["method"] for row in rows))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    x = np.arange(len(scenarios))
    width = min(0.8 / max(1, len(methods)), 0.18)
    for idx, method in enumerate(methods):
        loss_means = []
        latency_means = []
        for scenario in scenarios:
            method_rows = [row for row in rows if row["method"] == method and row["scenario"] == scenario]
            loss_means.append(np.mean([_float(row, "data_loss_rate") * 100.0 for row in method_rows]))
            latency_means.append(np.mean([_float(row, "recovery_latency_ms") for row in method_rows]))
        offset = (idx - (len(methods) - 1) / 2) * width
        label_row = next(row for row in rows if row["method"] == method)
        label = f"{label_row['baseline_id']} {label_row['baseline_name']}"
        color = E6_METHOD_COLORS.get(method, C["plain"])
        axes[0].bar(x + offset, loss_means, width, label=label, color=color)
        axes[1].bar(x + offset, latency_means, width, label=label, color=color)
    for ax, ylabel, title in [
        (axes[0], "Mean data loss rate (%)", "E6a - Analytical data loss by method"),
        (axes[1], "Recovery latency (ms)", "E6b - Modeled recovery latency by method"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    overhead_x = np.arange(len(methods))
    overhead_width = 0.36
    message_overhead = [
        np.mean([_float(row, "message_overhead") for row in rows if row["method"] == method])
        for method in methods
    ]
    compute_overhead = [
        np.mean([_float(row, "compute_overhead") for row in rows if row["method"] == method])
        for method in methods
    ]
    axes[2].bar(overhead_x - overhead_width / 2, message_overhead, overhead_width, label="Message", color=C["aes"])
    axes[2].bar(overhead_x + overhead_width / 2, compute_overhead, overhead_width, label="Compute", color=C["paillier"])
    axes[2].axhline(1.0, color="black", linestyle="--", lw=1, alpha=0.5)
    axes[2].set_xticks(overhead_x)
    axes[2].set_xticklabels(methods, rotation=15, ha="right")
    axes[2].set_ylabel("Mean overhead factor vs baseline")
    axes[2].set_title("E6c - Analytical overhead by method")
    axes[2].legend(fontsize=8)
    plt.tight_layout()
    _finish_figure(results_dir, "e6_fault_detection", show)


def plot_e6_tradeoff(results_dir: str = RESULTS_DIR, show: bool = False) -> None:
    rows = read_csv(os.path.join(results_dir, "e6_fault_detection.csv"))
    methods = list(dict.fromkeys(row["method"] for row in rows))
    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        mean_loss_pct = np.mean([_float(row, "data_loss_rate") * 100.0 for row in method_rows])
        mean_message_overhead = np.mean([_float(row, "message_overhead") for row in method_rows])
        mean_latency = np.mean([_float(row, "recovery_latency_ms") for row in method_rows])
        marker_size = 90.0 + mean_latency * 0.22
        is_proposed = method == "proposed_ack_kmm"
        ax.scatter(
            mean_message_overhead,
            mean_loss_pct,
            s=marker_size,
            color=E6_METHOD_COLORS.get(method, C["plain"]),
            edgecolor="black" if is_proposed else "white",
            linewidth=1.4 if is_proposed else 0.8,
            alpha=0.92,
            zorder=3 if is_proposed else 2,
        )
        label = E6_METHOD_LABELS.get(method, method)
        label_offsets = {
            "b1_gossip": (6, -15, "left"),
            "b2_replication": (-8, 12, "right"),
            "checkpoint": (8, 10, "left"),
            "b4_multilayer": (8, 2, "left"),
            "b5_fog_clustering": (8, 4, "left"),
            "proposed_ack_kmm": (8, -14, "left"),
        }
        x_offset, y_offset, h_align = label_offsets.get(method, (8, 4, "left"))
        ax.annotate(
            label,
            (mean_message_overhead, mean_loss_pct),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=9,
            weight="bold" if is_proposed else "normal",
            ha=h_align,
            va="center",
        )

    ax.axvline(1.0, color="black", linestyle="--", lw=1, alpha=0.45)
    ax.set_xlabel("Mean message overhead factor")
    ax.set_ylabel("Mean data loss rate (%)")
    ax.set_title("E6 - Modeled fault-recovery loss/overhead trade-off")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0.98, 2.08)
    ax.set_ylim(0.0, 82.0)
    ax.text(
        2.05,
        77.5,
        "Bubble area scales with recovery latency",
        fontsize=8,
        color="#555555",
        ha="right",
        va="top",
    )
    plt.tight_layout()
    _finish_figure(results_dir, "e6_fault_tradeoff", show)


def generate_all(results_dir: str = RESULTS_DIR, show: bool = False) -> None:
    plot_e1(results_dir, show=show)
    plot_e2(results_dir, show=show)
    plot_e3a(results_dir, show=show)
    plot_e5(results_dir, show=show)
    plot_e5_sensitivity(results_dir, show=show)


def generate_p2_all(results_dir: str = RESULTS_DIR, show: bool = False) -> None:
    plot_e3b(results_dir, show=show)
    plot_e4(results_dir, show=show)
    plot_e6(results_dir, show=show)
    plot_e6_tradeoff(results_dir, show=show)


def plot_e7(results_dir: str = RESULTS_DIR, show: bool = False) -> None:
    rows = read_csv(os.path.join(results_dir, "e7_pipeline_latency_summary.csv"))
    methods = [row["method"] for row in rows]
    labels = [METHOD_LABELS.get(method, method) for method in methods]
    totals = [_float(row, "total_ms_median") for row in rows]
    storage = [_float(row, "storage_items_per_window_median") for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    bars = ax.bar(labels, totals, color=[C["plain"], C["aes"], C["paillier"], C["yours"]], edgecolor="white")
    ax.axhline(WINDOW_MS, color="red", linestyle="--", lw=1, alpha=0.7, label="500 ms window")
    for bar, value in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Median end-to-end latency (ms)")
    ax.set_title("E7a - Pipeline latency by method")
    ax.tick_params(axis="x", labelrotation=12)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.bar(labels, storage, color=[C["plain"], C["aes"], C["paillier"], C["yours"]], edgecolor="white")
    ax2.set_yscale("log")
    ax2.set_ylabel("Median storage/window (items, log scale)")
    ax2.set_title("E7b - Stored values or ciphertexts per window")
    ax2.tick_params(axis="x", labelrotation=12)
    plt.tight_layout()
    _finish_figure(results_dir, "e7_pipeline_latency", show)


def plot_e8(results_dir: str = RESULTS_DIR, show: bool = False) -> None:
    rows = read_csv(os.path.join(results_dir, "e8_blast_radius.csv"))
    scenarios = [row["scenario"] for row in rows]
    x = np.arange(len(rows))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width / 2, [_float(row, "global_key_exposed_pct") for row in rows], width, label="Global key", color=C["paillier"], edgecolor="white")
    ax.bar(x + width / 2, [_float(row, "fog_scoped_exposed_pct") for row in rows], width, label="Fog-scoped keys", color=C["yours"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha="right")
    ax.set_ylabel("Sensors exposed (%)")
    ax.set_title("E8 - Analytical Blast Radius: Global vs Fog-Scoped Keys")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _finish_figure(results_dir, "e8_blast_radius", show)


def generate_p3_all(results_dir: str = RESULTS_DIR, show: bool = False) -> None:
    plot_e7(results_dir, show=show)
    plot_e8(results_dir, show=show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate P1 figures from CSV results")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--show", action="store_true", help="Display figures with matplotlib after saving")
    args = parser.parse_args()
    generate_all(args.results_dir, show=args.show)
