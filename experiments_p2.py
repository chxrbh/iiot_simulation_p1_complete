"""Experiment runners for P2 experiments: E3b, E4, and E6."""

from __future__ import annotations

from collections.abc import Callable
import random
import statistics
import time

from config import (
    E3B_BACKUP,
    E3B_BACKUP_OWN_N,
    E3B_K_DELEGATED_PER_SOURCE,
    E3B_SOURCE_N,
    E3B_SOURCES,
    E3B_TRIALS,
    E4_K_VALUES,
    E4_REPS,
    E6_CHECKPOINT_INTERVAL_MS,
    E6_CHECKPOINT_RESTORE_MS,
    E6_CLUSTER_DETECT_MS,
    E6_CLUSTER_REROUTE_MS,
    E6_CLUSTER_SELECTION_MS,
    E6_FAILURE_SCENARIOS,
    E6_GOSSIP_DETECT_MS,
    E6_METHODS,
    E6_MULTILAYER_DETECTION_MS,
    E6_SEEDS,
    FOG_NODES,
    KMM_PROV_MS,
    SCALE,
    T_ACK_MS,
    WINDOW_MS,
)
from crypto_sim import (
    KMM,
    aes_encrypt,
    paillier_ciphertext_bytes,
    paillier_encrypt,
    sgx_enclave_process,
    sgx_enclave_storage_prep,
)


def _readings(n: int, rng: random.Random) -> list[float]:
    return [round(rng.uniform(0, 150), 4) for _ in range(n)]


def run_e3b(
    pub_key: object,
    priv_key: object,
    k_fog: dict[str, bytes],
    k_store: bytes,
    seed: int,
    trials: int = E3B_TRIALS,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    rng = random.Random(seed + 3300)
    correct_scaled = 0
    quant_errors = []
    trial_times = []
    max_scaled_error = 0.0
    provisioned_edges: set[str] = set()

    for trial in range(trials):
        if progress:
            progress(f"E3b trial={trial + 1}/{trials}")
        t0 = time.perf_counter()
        kmm = KMM(k_fog)
        source_readings = {source: _readings(E3B_SOURCE_N, rng) for source in E3B_SOURCES}
        backup_readings = _readings(E3B_BACKUP_OWN_N, rng)
        true_float_sum = sum(sum(values) for values in source_readings.values()) + sum(backup_readings)
        true_scaled_sum = (
            sum(sum(int(value * SCALE) for value in values) for values in source_readings.values())
            + sum(int(value * SCALE) for value in backup_readings)
        ) / SCALE
        aggregates = {source: paillier_encrypt(pub_key, 0, rng) for source in E3B_SOURCES}
        aggregates[E3B_BACKUP] = paillier_encrypt(pub_key, 0, rng)

        for source, readings in source_readings.items():
            delegated = set(rng.sample(range(E3B_SOURCE_N), E3B_K_DELEGATED_PER_SOURCE))
            delegated_key, _ = kmm.provision_key(source, E3B_BACKUP)
            provisioned_edges.add(f"{source}->{E3B_BACKUP}")
            for idx, value in enumerate(readings):
                nonce, ct = aes_encrypt(k_fog[source], value, rng)
                if idx in delegated:
                    aggregates[source] = aggregates[source] + paillier_encrypt(pub_key, 0, rng)
                    aggregates[E3B_BACKUP] = aggregates[E3B_BACKUP] + sgx_enclave_process(
                        delegated_key, nonce, ct, pub_key, rng
                    )
                else:
                    aggregates[source] = aggregates[source] + sgx_enclave_process(
                        k_fog[source], nonce, ct, pub_key, rng
                    )

        for value in backup_readings:
            nonce, ct = aes_encrypt(k_fog[E3B_BACKUP], value, rng)
            aggregates[E3B_BACKUP] = aggregates[E3B_BACKUP] + sgx_enclave_process(
                k_fog[E3B_BACKUP], nonce, ct, pub_key, rng
            )

        final, _ = kmm.combine(aggregates, pub_key)
        _, _, decoded = sgx_enclave_storage_prep(final, priv_key, k_store, rng)
        for source in E3B_SOURCES:
            kmm.revoke_key(source, E3B_BACKUP)
        scaled_error = abs(decoded - true_scaled_sum)
        float_error = abs(decoded - true_float_sum)
        correct_scaled += int(scaled_error < 1e-9)
        max_scaled_error = max(max_scaled_error, scaled_error)
        quant_errors.append(float_error)
        trial_times.append((time.perf_counter() - t0) * 1000.0)

    total_sensors = E3B_SOURCE_N * len(E3B_SOURCES) + E3B_BACKUP_OWN_N
    return [
        {
            "sources": "+".join(E3B_SOURCES),
            "backup": E3B_BACKUP,
            "source_sensors_each": E3B_SOURCE_N,
            "backup_own_sensors": E3B_BACKUP_OWN_N,
            "total_sensors": total_sensors,
            "k_delegated_per_source": E3B_K_DELEGATED_PER_SOURCE,
            "trials": trials,
            "correct_scaled": correct_scaled,
            "accuracy_scaled_pct": correct_scaled / trials * 100.0,
            "median_quantization_error": statistics.median(quant_errors),
            "max_quantization_error": max(quant_errors),
            "max_scaled_error": max_scaled_error,
            "median_trial_ms": statistics.median(trial_times),
            "provisioned_edges": ";".join(sorted(provisioned_edges)),
            "security_note": "multi-source zero-fill uses randomized Paillier; indistinguishability is argued, not empirically tested",
        }
    ]


def run_e4(
    pub_key: object,
    seed: int,
    reps: int = E4_REPS,
    k_values: list[int] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    rng = random.Random(seed + 4000)
    k_values = k_values or E4_K_VALUES
    rows = []
    ct_bytes = paillier_ciphertext_bytes(pub_key)
    for k in k_values:
        if k < 1:
            raise ValueError("E4 requires at least one fog aggregate")
        latencies = []
        for rep in range(reps):
            if progress:
                progress(f"E4 k={k} rep={rep + 1}/{reps}")
            aggregates = [paillier_encrypt(pub_key, rng.randint(0, 100_000), rng) for _ in range(k)]
            t0 = time.perf_counter()
            combined = aggregates[0]
            for ciphertext in aggregates[1:]:
                combined = combined + ciphertext
            latencies.append((time.perf_counter() - t0) * 1000.0)
        rows.append(
            {
                "k_fog_aggregates": k,
                "he_additions": k - 1,
                "combine_latency_ms_median": statistics.median(latencies),
                "combine_latency_ms_std": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                "bytes_received": k * ct_bytes,
                "ciphertext_bytes": ct_bytes,
                "within_500ms": statistics.median(latencies) <= WINDOW_MS,
                "note": "KMM combine timing excludes ciphertext generation, zero encryption, and Paillier decryption",
            }
        )
    return rows


def _e6_reading_times(seed: int, observation_ms: float) -> list[float]:
    rng = random.Random(seed + 6000)
    total_sensors = sum(node["sensors"] for node in FOG_NODES.values())
    windows = int(observation_ms // WINDOW_MS)
    return [
        window * WINDOW_MS + rng.uniform(0.0, WINDOW_MS)
        for window in range(windows)
        for _ in range(total_sensors)
    ]


def _e6_count_lost(reading_times: list[float], lost_start_ms: float, lost_end_ms: float) -> int:
    return sum(lost_start_ms <= timestamp < lost_end_ms for timestamp in reading_times)


def _e6_count_at_or_after(reading_times: list[float], start_ms: float) -> int:
    return sum(timestamp >= start_ms for timestamp in reading_times)


E6_BASELINE_METADATA = {
    "b1_gossip": {
        "baseline_id": "B1",
        "baseline_name": "Gossip-based Failure Detection",
        "baseline_model": "failure detected after missing r consecutive gossip updates, modeled as r*Tg",
    },
    "b2_replication": {
        "baseline_id": "B2",
        "baseline_name": "Replication-Based Fault Tolerance",
        "baseline_model": "each reading is sent to both primary and backup fog nodes; backup result is used after primary failure",
    },
    "checkpoint": {
        "baseline_id": "B3",
        "baseline_name": "Checkpoint-Based Recovery",
        "baseline_model": "periodic checkpoint/restart; state after latest checkpoint and before restore completion may be lost",
    },
    "b4_multilayer": {
        "baseline_id": "B4",
        "baseline_name": "Multilayer Fault Detection",
        "baseline_model": "failure can be detected by infrastructure-level and application-level monitoring layers",
    },
    "b5_fog_clustering": {
        "baseline_id": "B5",
        "baseline_name": "Fog-Clustering Fault Tolerance",
        "baseline_model": "cluster coordinator detects primary fog failure, selects a replacement fog, and reroutes readings",
    },
    "proposed_ack_kmm": {
        "baseline_id": "Proposed",
        "baseline_name": "ACK+KMM Fault Recovery",
        "baseline_model": "sensor ACK timeout followed by KMM key provisioning and retransmission to backup fog",
    },
}


def _e6_method_metrics(method: str, failure_time_ms: float, seed: int) -> dict[str, object]:
    observation_ms = WINDOW_MS + max(
        E6_GOSSIP_DETECT_MS,
        E6_CHECKPOINT_RESTORE_MS,
        E6_MULTILAYER_DETECTION_MS,
        E6_CLUSTER_DETECT_MS + E6_CLUSTER_SELECTION_MS + E6_CLUSTER_REROUTE_MS,
        T_ACK_MS + KMM_PROV_MS,
    )
    reading_times = _e6_reading_times(seed, observation_ms)
    total_readings = len(reading_times)
    observation_windows = int(observation_ms // WINDOW_MS)
    baseline_message_units = float(total_readings)
    baseline_compute_ops = float(total_readings)
    lost_start_ms = failure_time_ms
    lost_end_ms = failure_time_ms
    control_messages = 0.0
    kmm_provision_messages = 0.0
    checkpoint_writes = 0.0
    restored_readings = 0.0
    backup_rerouted_readings = 0.0
    if method == "b1_gossip":
        lost_end_ms = failure_time_ms + E6_GOSSIP_DETECT_MS
        recovery_latency_ms = E6_GOSSIP_DETECT_MS
        control_messages = 3.0 * len(FOG_NODES)
        message_units = baseline_message_units + control_messages
        compute_ops = baseline_compute_ops
    elif method == "b2_replication":
        recovery_latency_ms = 0.0
        message_units = baseline_message_units * 2.0
        compute_ops = baseline_compute_ops * 2.0
    elif method == "checkpoint":
        last_checkpoint = (failure_time_ms // E6_CHECKPOINT_INTERVAL_MS) * E6_CHECKPOINT_INTERVAL_MS
        lost_start_ms = last_checkpoint
        lost_end_ms = failure_time_ms + E6_CHECKPOINT_RESTORE_MS  # restore window: data unavailable until fully restored
        recovery_latency_ms = E6_CHECKPOINT_RESTORE_MS
        checkpoint_writes = float(observation_windows * len(FOG_NODES))
    elif method == "b4_multilayer":
        lost_end_ms = failure_time_ms + E6_MULTILAYER_DETECTION_MS
        recovery_latency_ms = E6_MULTILAYER_DETECTION_MS
        control_messages = float(observation_windows * len(FOG_NODES) * 2)
        message_units = baseline_message_units + control_messages
        compute_ops = baseline_compute_ops + control_messages
    elif method == "b5_fog_clustering":
        recovery_latency_ms = E6_CLUSTER_DETECT_MS + E6_CLUSTER_SELECTION_MS + E6_CLUSTER_REROUTE_MS
        lost_end_ms = failure_time_ms + recovery_latency_ms
        control_messages = float(observation_windows * len(FOG_NODES) + 3)
        backup_rerouted_readings = float(_e6_count_at_or_after(reading_times, lost_end_ms))
        message_units = baseline_message_units + control_messages
        compute_ops = baseline_compute_ops + control_messages + 1.0
    elif method == "proposed_ack_kmm":
        lost_end_ms = failure_time_ms + T_ACK_MS + KMM_PROV_MS
        recovery_latency_ms = T_ACK_MS + KMM_PROV_MS
        control_messages = float(total_readings)
        kmm_provision_messages = 2.0
        backup_rerouted_readings = float(_e6_count_at_or_after(reading_times, lost_end_ms))
        message_units = baseline_message_units + control_messages * 0.05 + kmm_provision_messages
        compute_ops = baseline_compute_ops + kmm_provision_messages
    else:
        raise ValueError(f"Unknown E6 method: {method}")
    lost_readings = _e6_count_lost(reading_times, lost_start_ms, lost_end_ms)
    recovered_readings = total_readings - lost_readings
    if method == "checkpoint":
        restored_readings = float(lost_readings)
        message_units = baseline_message_units + checkpoint_writes
        compute_ops = baseline_compute_ops + checkpoint_writes + restored_readings
    data_loss_rate = lost_readings / total_readings
    return {
        "total_readings": total_readings,
        "recovered_readings": recovered_readings,
        "lost_readings": lost_readings,
        "data_loss_rate": data_loss_rate,
        "window_completeness": recovered_readings / total_readings,
        "recovery_latency_ms": recovery_latency_ms,
        "baseline_message_units": baseline_message_units,
        "measured_message_units": message_units,
        "control_messages": control_messages,
        "kmm_provision_messages": kmm_provision_messages,
        "checkpoint_writes": checkpoint_writes,
        "backup_rerouted_readings": backup_rerouted_readings,
        "baseline_compute_ops": baseline_compute_ops,
        "measured_compute_ops": compute_ops,
        "restored_readings": restored_readings,
        "message_overhead": message_units / baseline_message_units,
        "compute_overhead": compute_ops / baseline_compute_ops,
    }


def run_e6(progress: Callable[[str], None] | None = None) -> list[dict[str, object]]:
    rows = []
    for scenario, failure_time_ms in E6_FAILURE_SCENARIOS.items():
        for seed in E6_SEEDS:
            for method in E6_METHODS:
                if progress:
                    progress(f"E6 scenario={scenario} seed={seed} method={method}")
                metrics = _e6_method_metrics(method, failure_time_ms, seed)
                metadata = E6_BASELINE_METADATA[method]
                rows.append(
                    {
                        "method": method,
                        **metadata,
                        "scenario": scenario,
                        "fail_time_ms": int(failure_time_ms),
                        "seed": seed,
                        **metrics,
                    }
                )
    return rows
