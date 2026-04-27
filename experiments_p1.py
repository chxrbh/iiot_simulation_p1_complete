"""Experiment runners for the repaired P1 package."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
import random
import statistics
import time
from collections.abc import Callable

from config import (
    BASE_WORKLOAD,
    E1_N_VALUES,
    E2_N_VALUES,
    E2_REPS,
    E3_K_VALUES,
    E3_N,
    E3_TRIALS,
    E5_TASKS_PER_WINDOW,
    E5_TO_RATIO,
    FOG_NODES,
    SCALE,
    SEEDS,
    TAU,
    WINDOW_MS,
    build_schedule,
    capacity_score,
)
from crypto_sim import (
    KMM,
    PaillierPrivateKey,
    PaillierPublicKey,
    aes_decrypt,
    aes_encrypt,
    generate_fog_keys,
    paillier_encrypt,
    paillier_ciphertext_bytes,
    sgx_enclave_process,
    sgx_enclave_storage_prep,
)

CapacityWeights = dict[str, tuple[float, float, float]]

DEFAULT_CAPACITY_WEIGHTS: CapacityWeights = {
    "LS": (0.20, 0.50, 0.30),
    "TO": (0.50, 0.15, 0.35),
}

CAPACITY_WEIGHT_VARIANTS: dict[str, CapacityWeights] = {
    "configured": DEFAULT_CAPACITY_WEIGHTS,
    "balanced": {
        "LS": (1 / 3, 1 / 3, 1 / 3),
        "TO": (1 / 3, 1 / 3, 1 / 3),
    },
    "load_heavy": {
        "LS": (0.60, 0.20, 0.20),
        "TO": (0.60, 0.20, 0.20),
    },
    "latency_heavy": {
        "LS": (0.20, 0.60, 0.20),
        "TO": (0.20, 0.60, 0.20),
    },
}


def run_e1(pub_key: PaillierPublicKey) -> list[dict[str, object]]:
    paillier_bytes = paillier_ciphertext_bytes(pub_key)
    # AES-GCM per-reading byte breakdown:
    #   nonce:  12 bytes (96-bit, NIST SP 800-38D)
    #   body:    8 bytes = len("149.9999") — worst-case ASCII repr of
    #            round(uniform(0, 150), 4); AES-GCM is a stream cipher so
    #            ciphertext body length equals plaintext length exactly.
    #   tag:    16 bytes (128-bit GCM authentication tag)
    aes_bytes_per_reading = 12 + len("149.9999") + 16  # = 36
    plaintext_bytes_per_reading = 8
    rows = []
    for n in E1_N_VALUES:
        aes_bytes = n * aes_bytes_per_reading
        pnb_bytes = n * paillier_bytes
        ours_bytes_paillier = paillier_bytes
        ours_bytes_cloud = aes_bytes_per_reading
        rows.append(
            {
                "n": n,
                "plaintext_values": n,
                "plaintext_bytes": n * plaintext_bytes_per_reading,
                "aes_ciphertexts": n,
                "aes_bytes": aes_bytes,
                "paillier_nobatch_ciphertexts": n,
                "paillier_nobatch_bytes": pnb_bytes,
                "ours_ciphertexts": 1,
                "ours_paillier_bytes": ours_bytes_paillier,
                "ours_cloud_aes_bytes": ours_bytes_cloud,
                "ciphertext_count_reduction": n,
                "byte_reduction_vs_aes": aes_bytes / ours_bytes_cloud,
                "byte_reduction_vs_paillier_nobatch": pnb_bytes / ours_bytes_paillier,
            }
        )
    return rows


def _readings(n: int, rng: random.Random) -> list[float]:
    return [round(rng.uniform(0, 150), 4) for _ in range(n)]


def _stats(values: list[float]) -> tuple[float, float]:
    return statistics.median(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _time_plaintext(readings: list[float]) -> float:
    t0 = time.perf_counter()
    _ = sum(readings)
    return (time.perf_counter() - t0) * 1000.0


def _time_aes(readings: list[float], key: bytes, rng: random.Random) -> float:
    t0 = time.perf_counter()
    pairs = [aes_encrypt(key, value, rng) for value in readings]
    _ = sum(aes_decrypt(key, nonce, ct) for nonce, ct in pairs)
    return (time.perf_counter() - t0) * 1000.0


def _time_paillier_nobatch(
    readings: list[float],
    key: bytes,
    pub_key: PaillierPublicKey,
    priv_key: PaillierPrivateKey,
    rng: random.Random,
) -> float:
    t0 = time.perf_counter()
    pairs = [aes_encrypt(key, value, rng) for value in readings]
    ciphertexts = [sgx_enclave_process(key, nonce, ct, pub_key, rng) for nonce, ct in pairs]
    cloud_aggregate = paillier_encrypt(pub_key, 0, rng)
    for ciphertext in ciphertexts:
        cloud_aggregate = cloud_aggregate + ciphertext
    _ = priv_key.decrypt(cloud_aggregate) / SCALE
    return (time.perf_counter() - t0) * 1000.0


def _time_ours(
    readings: list[float],
    key: bytes,
    store_key: bytes,
    pub_key: PaillierPublicKey,
    priv_key: PaillierPrivateKey,
    rng: random.Random,
) -> tuple[float, float, float, float]:
    kmm = KMM({"F1": key})
    t0 = time.perf_counter()
    agg = paillier_encrypt(pub_key, 0, rng)
    t_enc0 = time.perf_counter()
    for value in readings:
        nonce, ct = aes_encrypt(key, value, rng)
        agg = agg + sgx_enclave_process(key, nonce, ct, pub_key, rng)
    enc_ms = (time.perf_counter() - t_enc0) * 1000.0
    combined, kmm_ms = kmm.combine({"F1": agg}, pub_key)
    t_store0 = time.perf_counter()
    _ = sgx_enclave_storage_prep(combined, priv_key, store_key, rng)
    storage_ms = (time.perf_counter() - t_store0) * 1000.0
    total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms, enc_ms, kmm_ms, storage_ms


def run_e2(
    pub_key: PaillierPublicKey,
    priv_key: PaillierPrivateKey,
    k_fog: dict[str, bytes],
    k_store: bytes,
    seed: int,
    reps: int = E2_REPS,
    n_values: list[int] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    rng = random.Random(seed + 2000)
    n_values = n_values or E2_N_VALUES
    rows = []
    for n in n_values:
        timings = defaultdict(list)
        for rep in range(reps):
            if progress:
                progress(f"E2 n={n} rep={rep + 1}/{reps}")
            readings = _readings(n, rng)
            timings["plaintext_ms"].append(_time_plaintext(readings))
            timings["aes_ms"].append(_time_aes(readings, k_fog["F1"], rng))
            timings["paillier_nobatch_ms"].append(
                _time_paillier_nobatch(readings, k_fog["F1"], pub_key, priv_key, rng)
            )
            total, enc, kmm, storage = _time_ours(readings, k_fog["F1"], k_store, pub_key, priv_key, rng)
            timings["ours_ms"].append(total)
            timings["ours_enclave_ms"].append(enc)
            timings["ours_kmm_ms"].append(kmm)
            timings["ours_storage_ms"].append(storage)
        row: dict[str, object] = {"n": n, "window_ms": WINDOW_MS}
        for name, values in timings.items():
            med, std = _stats(values)
            row[f"{name}_median"] = med
            row[f"{name}_std"] = std
        row["ours_within_500ms"] = row["ours_ms_median"] <= WINDOW_MS
        row["conclusion"] = (
            "violates_500ms_window" if row["ours_ms_median"] > WINDOW_MS else "within_500ms_window"
        )
        rows.append(row)
    return rows


def run_e3a(
    pub_key: PaillierPublicKey,
    priv_key: PaillierPrivateKey,
    k_fog: dict[str, bytes],
    k_store: bytes,
    seed: int,
    trials: int = E3_TRIALS,
    n: int = E3_N,
    k_values: list[int] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    rng = random.Random(seed + 3000)
    k_values = k_values or E3_K_VALUES
    rows = []
    for k in k_values:
        correct_scaled = 0
        quant_errors = []
        trial_times = []
        max_scaled_error = 0.0
        for trial in range(trials):
            if progress:
                progress(f"E3a k={k} trial={trial + 1}/{trials}")
            t0 = time.perf_counter()
            readings = _readings(n, rng)
            true_float_sum = sum(readings)
            true_scaled_sum = sum(int(value * SCALE) for value in readings) / SCALE
            delegated = set(rng.sample(range(n), k))
            kmm = KMM(k_fog)
            delegated_key, _ = kmm.provision_key("F1", "F4")
            agg_a = paillier_encrypt(pub_key, 0, rng)
            agg_b = paillier_encrypt(pub_key, 0, rng)
            for i, value in enumerate(readings):
                nonce, ct = aes_encrypt(k_fog["F1"], value, rng)
                if i in delegated:
                    agg_a = agg_a + paillier_encrypt(pub_key, 0, rng)
                    agg_b = agg_b + sgx_enclave_process(delegated_key, nonce, ct, pub_key, rng)
                else:
                    agg_a = agg_a + sgx_enclave_process(k_fog["F1"], nonce, ct, pub_key, rng)
            final, _ = kmm.combine({"F1": agg_a, "F4": agg_b}, pub_key)
            _, _, decoded = sgx_enclave_storage_prep(final, priv_key, k_store, rng)
            kmm.revoke_key("F1", "F4")
            scaled_error = abs(decoded - true_scaled_sum)
            float_error = abs(decoded - true_float_sum)
            max_scaled_error = max(max_scaled_error, scaled_error)
            quant_errors.append(float_error)
            correct_scaled += int(scaled_error < 1e-9)
            trial_times.append((time.perf_counter() - t0) * 1000.0)
        rows.append(
            {
                "n": n,
                "k_delegated": k,
                "trials": trials,
                "correct_scaled": correct_scaled,
                "accuracy_scaled_pct": correct_scaled / trials * 100.0,
                "median_quantization_error": statistics.median(quant_errors),
                "max_quantization_error": max(quant_errors),
                "max_scaled_error": max_scaled_error,
                "median_trial_ms": statistics.median(trial_times),
                "zero_fill_security_note": (
                    "randomized Paillier zero-fill; indistinguishability argued cryptographically, "
                    "not experimentally tested"
                ),
            }
        )
    return rows


class RoundRobin:
    def __init__(self) -> None:
        self.index = 0

    def pick(self, candidates: list[str], task_type: str, wt: dict[str, dict[str, float]], rng: random.Random) -> str | None:
        if not candidates:
            return None
        choice = candidates[self.index % len(candidates)]
        self.index += 1
        return choice


def _capacity_score_weighted(
    node_id: str,
    wt: dict[str, dict[str, float]],
    task_type: str,
    capacity_weights: CapacityWeights | None = None,
) -> float:
    if capacity_weights is None:
        return capacity_score(node_id, wt, task_type)
    w1, w2, w3 = capacity_weights[task_type]
    node = wt[node_id]
    return w1 * (1 - node["workload"]) + w2 * (1 - node["latency"]) + w3 * (1 - node["queue"])


def _pick(
    method: str,
    candidates: list[str],
    task_type: str,
    wt: dict[str, dict[str, float]],
    rng: random.Random,
    rr: RoundRobin,
    capacity_weights: CapacityWeights | None = None,
) -> str | None:
    if not candidates:
        return None
    if method == "random":
        return rng.choice(candidates)
    if method == "round_robin":
        return rr.pick(candidates, task_type, wt, rng)
    if method == "threshold":
        return min(candidates, key=lambda nid: wt[nid]["workload"])
    if method == "capacity":
        return max(candidates, key=lambda nid: _capacity_score_weighted(nid, wt, task_type, capacity_weights))
    raise ValueError(f"Unknown E5 method: {method}")


def _service_time(node_id: str, task_type: str, wt: dict[str, dict[str, float]]) -> float:
    node = FOG_NODES[node_id]
    base = 28.0 if task_type == "LS" else 90.0
    cpu_penalty = 4.0 / node["cpu_cap"]
    latency_penalty = wt[node_id]["latency"] * 60.0
    queue_penalty = wt[node_id]["queue"] * (70.0 if task_type == "LS" else 120.0)
    return base * cpu_penalty + latency_penalty + queue_penalty


def _run_e5_once(
    method: str,
    seed: int,
    capacity_weights: CapacityWeights | None = None,
    weight_variant: str = "",
) -> dict[str, object]:
    rng = random.Random(seed)
    current = {nid: dict(values) for nid, values in BASE_WORKLOAD.items()}
    rr = RoundRobin()
    schedule = build_schedule()
    total_tasks = completed = deadline_met = redelegated = 0
    delegated_tasks = 0
    ls_delegated = ls_correct = to_delegated = to_correct = 0
    stdevs = []
    assignments = defaultdict(int)
    for window in schedule:
        for nid in ["F1", "F3", "F4", "F5"]:
            current[nid]["workload"] = max(
                current[nid]["workload"] * 0.85 + BASE_WORKLOAD[nid]["workload"] * 0.15,
                BASE_WORKLOAD[nid]["workload"],
            )
            current[nid]["queue"] = max(
                current[nid]["queue"] * 0.85 + BASE_WORKLOAD[nid]["queue"] * 0.15,
                BASE_WORKLOAD[nid]["queue"],
            )
        current["F2"]["workload"] = float(window["f2_load"])
        current["F2"]["queue"] = float(window["f2_load"]) * 0.8
        if not window["f5_alive"]:
            current["F5"]["workload"] = 1.0
            current["F5"]["queue"] = 1.0
        overloaded = {nid for nid, values in current.items() if values["workload"] >= TAU}
        wt_snapshot = {nid: dict(values) for nid, values in current.items()}
        stdevs.append(statistics.pstdev(values["workload"] for values in wt_snapshot.values()))
        for task_idx in range(E5_TASKS_PER_WINDOW):
            task_type = "TO" if rng.random() < E5_TO_RATIO else "LS"
            source = "F2" if current["F2"]["workload"] >= TAU else f"F{1 + (task_idx % 5)}"
            delegated_task = source in overloaded
            if not delegated_task:
                target = source
            else:
                delegated_tasks += 1
                if task_type == "LS":
                    ls_delegated += 1
                else:
                    to_delegated += 1
                candidates = [
                    nid
                    for nid, values in wt_snapshot.items()
                    if nid not in overloaded and values["workload"] < TAU
                ]
                target = _pick(method, candidates, task_type, wt_snapshot, rng, rr, capacity_weights)
            total_tasks += 1
            if target is None:
                if delegated_task:
                    redelegated += 1
                continue
            assignments[target] += 1
            latency = _service_time(target, task_type, current)
            deadline = 100.0 if task_type == "LS" else 1000.0
            is_weak_overload = delegated_task and target == "F3"
            if is_weak_overload:
                redelegated += 1
            else:
                completed += 1
                deadline_met += int(latency <= deadline)
            if delegated_task and task_type == "LS":
                ls_correct += int(target == "F4")
            elif delegated_task and task_type == "TO":
                to_correct += int(target == "F1")
            if delegated_task:
                current[target]["workload"] = min(0.99, current[target]["workload"] + 0.20 / E5_TASKS_PER_WINDOW)
                current[target]["queue"] = min(0.99, current[target]["queue"] + 0.12 / E5_TASKS_PER_WINDOW)
    return {
        "method": method,
        "seed": seed,
        "tasks": total_tasks,
        "delegated_tasks": delegated_tasks,
        "completion_pct": completed / total_tasks * 100.0,
        "delegated_completion_pct": (delegated_tasks - redelegated) / max(1, delegated_tasks) * 100.0,
        "deadline_satisfaction_pct": deadline_met / total_tasks * 100.0,
        "redelegation_rate_pct": redelegated / max(1, delegated_tasks) * 100.0,
        "ls_capacity_score_agreement_pct": ls_correct / max(1, ls_delegated) * 100.0,
        "oracle_is_capacity_score_winner": True,  # LS oracle = CapacityScore winner at base load = F4
        "to_policy_agreement_pct": to_correct / max(1, to_delegated) * 100.0,
        "workload_stdev": statistics.mean(stdevs),
        "assignment_f1": assignments["F1"],
        "assignment_f3": assignments["F3"],
        "assignment_f4": assignments["F4"],
        "weight_variant": weight_variant,
    }


# Two-tailed t critical values for 95% CI, keyed by degrees of freedom (df = n-1).
# For df >= 30 the normal approximation z=1.96 is used (error < 0.1%).
# Values from standard t-distribution tables (matched to scipy.stats.t.ppf(0.975, df)).
_T95_TABLE: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    20: 2.086, 29: 2.045,
}


def _t95(n: int) -> float:
    """Return the two-tailed t critical value for a 95% CI with n observations.

    Uses exact table values for df in {1..10, 20, 29}, linear interpolation for
    df in (10, 29), and the normal approximation 1.96 for df >= 30.
    """
    df = max(n - 1, 1)
    if df in _T95_TABLE:
        return _T95_TABLE[df]
    if df >= 30:
        return 1.96
    sorted_keys = sorted(_T95_TABLE)
    lo = max(k for k in sorted_keys if k < df)
    hi = min(k for k in sorted_keys if k > df)
    return _T95_TABLE[lo] + (df - lo) / (hi - lo) * (_T95_TABLE[hi] - _T95_TABLE[lo])


def run_e5(
    seeds: list[int] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    seeds = seeds or SEEDS
    methods = ["random", "round_robin", "threshold", "capacity"]
    runs = []
    for method in methods:
        for seed in seeds:
            if progress:
                progress(f"E5 method={method} seed={seed}")
            runs.append(_run_e5_once(method, seed))
    aggregate = []
    for method in methods:
        method_runs = [row for row in runs if row["method"] == method]
        row: dict[str, object] = {"method": method, "runs": len(method_runs)}
        for metric in [
            "completion_pct",
            "deadline_satisfaction_pct",
            "redelegation_rate_pct",
            "ls_capacity_score_agreement_pct",
            "to_policy_agreement_pct",
            "workload_stdev",
        ]:
            values = [float(run[metric]) for run in method_runs]
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            ci95 = _t95(len(values)) * std / sqrt(len(values)) if values else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci95
        row["baseline_note"] = "heuristic baseline; no claim of state-of-the-art tuning"
        row["policy_agreement_note"] = (
            "ls_capacity_score_agreement_pct measures agreement with the CapacityScore "
            "winner at base workload (F4 for LS, F1 for TO). This is circular for the "
            "capacity method; it measures divergence for heuristic baselines."
        )
        aggregate.append(row)
    return runs, aggregate


def run_e5_sensitivity(seeds: list[int] | None = None) -> list[dict[str, object]]:
    seeds = seeds or SEEDS
    rows = []
    for variant, weights in CAPACITY_WEIGHT_VARIANTS.items():
        runs = [_run_e5_once("capacity", seed, capacity_weights=weights, weight_variant=variant) for seed in seeds]
        row: dict[str, object] = {"method": "capacity", "weight_variant": variant, "runs": len(runs)}
        for metric in [
            "completion_pct",
            "deadline_satisfaction_pct",
            "redelegation_rate_pct",
            "ls_capacity_score_agreement_pct",
            "to_policy_agreement_pct",
            "workload_stdev",
        ]:
            values = [float(run[metric]) for run in runs]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        row["interpretation_note"] = "CapacityScore weight sensitivity; variants are heuristic, not tuned on a validation set"
        rows.append(row)
    return rows
