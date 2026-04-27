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
    E6_ACK_KEY_BYTES,
    E6_CHECKPOINT_INTERVAL_MS,
    E6_FAILURE_SCENARIOS,
    E6_GOSSIP_DETECT_MS,
    E6_METHODS,
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


def _e6_method_metrics(method: str, failure_time_ms: float) -> dict[str, object]:
    remaining_ms = max(0.0, WINDOW_MS - failure_time_ms)
    total_sensors = sum(node["sensors"] for node in FOG_NODES.values())
    if method == "none":
        data_loss_ms = remaining_ms
        recovery_latency_ms = E6_GOSSIP_DETECT_MS
        storage_overhead_factor = 1.0
        storage_overhead_bytes = 0
        limitation = "all readings after failure are lost until manual/gossip recovery"
    elif method == "replication":
        data_loss_ms = 0.0
        recovery_latency_ms = 0.0
        storage_overhead_factor = 2.0
        storage_overhead_bytes = total_sensors * 44
        limitation = "models hot replication with 2x reading storage overhead"
    elif method == "checkpoint":
        data_loss_ms = min(remaining_ms, E6_CHECKPOINT_INTERVAL_MS)
        recovery_latency_ms = 500.0
        storage_overhead_factor = 1.25
        storage_overhead_bytes = total_sensors * 8
        limitation = "bounded by checkpoint interval; not instant recovery"
    elif method == "ack_kmm":
        data_loss_ms = min(remaining_ms, T_ACK_MS + KMM_PROV_MS)
        recovery_latency_ms = T_ACK_MS + KMM_PROV_MS
        storage_overhead_factor = 1.0
        storage_overhead_bytes = len(FOG_NODES) * E6_ACK_KEY_BYTES
        limitation = "deterministic timing model; readings during ACK and key provisioning are lost"
    else:
        raise ValueError(f"Unknown E6 method: {method}")
    completion_pct = (WINDOW_MS - data_loss_ms) / WINDOW_MS * 100.0
    return {
        "data_loss_ms": data_loss_ms,
        "completion_pct": completion_pct,
        "storage_overhead_factor": storage_overhead_factor,
        "storage_overhead_bytes": storage_overhead_bytes,
        "recovery_latency_ms": recovery_latency_ms,
        "limitation": limitation,
    }


def run_e6(progress: Callable[[str], None] | None = None) -> list[dict[str, object]]:
    rows = []
    for scenario, failure_time_ms in E6_FAILURE_SCENARIOS.items():
        for method in E6_METHODS:
            if progress:
                progress(f"E6 scenario={scenario} method={method}")
            metrics = _e6_method_metrics(method, failure_time_ms)
            rows.append(
                {
                    "scenario": scenario,
                    "failure_time_ms": failure_time_ms,
                    "method": method,
                    "window_ms": WINDOW_MS,
                    **metrics,
                    "model_note": "analytical timing model; not live distributed fault injection",
                }
            )
    return rows
