"""Experiment runners for P3 experiments: E7 and E8."""

from __future__ import annotations

from collections.abc import Callable
import random
import statistics

from config import (
    E7_CLOUD_UPLOAD_MS,
    E7_KMM_COMBINE_MS,
    E7_METHODS,
    E7_N,
    E7_NET_SENSOR_TO_CLOUD_MS,
    E7_NET_SENSOR_TO_FOG_MS,
    E7_NODE,
    E7_PAILLIER_ADD_MS,
    E7_PAILLIER_ENC_MS,
    E7_PLAINTEXT_SUM_MS,
    E7_SENSOR_AES_MS,
    E7_STAGE_COLUMNS,
    E7_STORAGE_PREP_MS,
    E7_TEE_DELEGATION_MS,
    E8_SCENARIOS,
    FOG_NODES,
    TAU,
    WINDOW_MS,
    build_schedule,
)
from crypto_sim import (
    paillier_ciphertext_bytes,
)


def _readings(n: int, rng: random.Random) -> list[float]:
    return [round(rng.uniform(0, 150), 4) for _ in range(n)]


def _stage_row_base(window: int, event: str, method: str, n: int, delegation_active: bool) -> dict[str, object]:
    return {
        "window": window,
        "event": event,
        "method": method,
        "n": n,
        "delegation_active": delegation_active,
        "window_ms": WINDOW_MS,
        **{column: 0.0 for column in E7_STAGE_COLUMNS},
        "privacy": False,
        "storage_items_per_window": 0,
        "storage_bytes_per_window": 0,
        "privacy_note": "",
    }


def _finish_e7_row(row: dict[str, object]) -> dict[str, object]:
    total = sum(float(row[column]) for column in E7_STAGE_COLUMNS)
    row["fog_latency_ms"] = sum(
        float(row[column])
        for column in [
            "plaintext_sum_ms",
            "enclave_aes_to_paillier_ms",
            "paillier_accum_ms",
            "delegation_ms",
            "kmm_combine_ms",
            "storage_prep_ms",
        ]
    )
    row["e2e_latency_ms"] = total
    row["total_ms"] = total
    row["within_500ms"] = total <= WINDOW_MS
    row["conclusion"] = "within_500ms_window" if total <= WINDOW_MS else "violates_500ms_window"
    return row


def _run_cloud_only(window: int, event: str, readings: list[float], n: int, delegation_active: bool) -> dict[str, object]:
    row = _stage_row_base(window, event, "cloud_only", n, delegation_active)
    row["sensor_to_cloud_ms"] = E7_NET_SENSOR_TO_CLOUD_MS
    row["cloud_upload_ms"] = E7_CLOUD_UPLOAD_MS
    row["storage_items_per_window"] = n
    row["storage_bytes_per_window"] = n * 8
    row["privacy_note"] = "cloud-centric baseline: no fog processing, no aggregation privacy, stores raw readings"
    return _finish_e7_row(row)


def _run_fog_plaintext(
    window: int,
    event: str,
    readings: list[float],
    key: bytes,
    rng: random.Random,
    n: int,
    delegation_active: bool,
) -> dict[str, object]:
    row = _stage_row_base(window, event, "fog_plaintext", n, delegation_active)
    row["sensor_to_fog_ms"] = E7_NET_SENSOR_TO_FOG_MS
    row["plaintext_sum_ms"] = E7_PLAINTEXT_SUM_MS
    row["cloud_upload_ms"] = E7_CLOUD_UPLOAD_MS
    row["storage_items_per_window"] = 1
    row["storage_bytes_per_window"] = 8
    row["privacy_note"] = "conventional fog aggregation lower bound: one plaintext sum, fog sees raw readings"
    return _finish_e7_row(row)


def _run_paillier_fog_convert(
    window: int,
    event: str,
    readings: list[float],
    key: bytes,
    store_key: bytes,
    pub_key: object,
    priv_key: object,
    rng: random.Random,
    n: int,
    delegation_active: bool,
) -> dict[str, object]:
    row = _stage_row_base(window, event, "paillier_fog_convert", n, delegation_active)
    row["sensor_to_fog_ms"] = E7_NET_SENSOR_TO_FOG_MS
    row["sensor_aes_ms"] = E7_SENSOR_AES_MS
    row["enclave_aes_to_paillier_ms"] = n * E7_PAILLIER_ENC_MS
    row["paillier_accum_ms"] = n * E7_PAILLIER_ADD_MS
    row["cloud_upload_ms"] = E7_CLOUD_UPLOAD_MS
    row["privacy"] = True
    row["storage_items_per_window"] = n
    row["storage_bytes_per_window"] = n * paillier_ciphertext_bytes(pub_key)
    row["privacy_note"] = "P2-SWAN-style Paillier baseline: fog converts and adds ciphertexts, stores n ciphertexts"
    return _finish_e7_row(row)


def _run_ours(
    window: int,
    event: str,
    readings: list[float],
    key: bytes,
    store_key: bytes,
    pub_key: object,
    priv_key: object,
    rng: random.Random,
    n: int,
    delegation_active: bool,
) -> dict[str, object]:
    row = _stage_row_base(window, event, "ours", n, delegation_active)
    row["sensor_to_fog_ms"] = E7_NET_SENSOR_TO_FOG_MS
    row["sensor_aes_ms"] = E7_SENSOR_AES_MS
    row["enclave_aes_to_paillier_ms"] = n * E7_PAILLIER_ENC_MS
    row["paillier_accum_ms"] = n * E7_PAILLIER_ADD_MS
    row["delegation_ms"] = E7_TEE_DELEGATION_MS if delegation_active else 0.0
    row["kmm_combine_ms"] = E7_KMM_COMBINE_MS
    row["storage_prep_ms"] = E7_STORAGE_PREP_MS
    row["cloud_upload_ms"] = E7_CLOUD_UPLOAD_MS
    row["privacy"] = True
    row["storage_items_per_window"] = 1
    row["storage_bytes_per_window"] = paillier_ciphertext_bytes(pub_key)
    row["privacy_note"] = "TEE AES-to-Paillier conversion with Paillier slot aggregation; delegation overhead only on delegated windows"
    return _finish_e7_row(row)


def run_e7(
    pub_key: object,
    priv_key: object,
    k_fog: dict[str, bytes],
    k_store: bytes,
    seed: int,
    n: int = E7_N,
    windows: int | None = None,
    methods: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(seed + 7000)
    schedule = build_schedule()
    if windows is not None:
        schedule = schedule[:windows]
    methods = methods or E7_METHODS
    rows = []
    for item in schedule:
        window = int(item["window"])
        event = str(item["event"])
        delegation_active = float(item["f2_load"]) >= TAU
        for method in methods:
            if progress:
                progress(f"E7 window={window} method={method}")
            readings = _readings(n, rng)
            if method == "cloud_only":
                row = _run_cloud_only(window, event, readings, n, delegation_active)
            elif method == "fog_plaintext":
                row = _run_fog_plaintext(window, event, readings, k_fog[E7_NODE], rng, n, delegation_active)
            elif method == "paillier_fog_convert":
                row = _run_paillier_fog_convert(
                    window, event, readings, k_fog[E7_NODE], k_store, pub_key, priv_key, rng, n, delegation_active
                )
            elif method == "ours":
                row = _run_ours(
                    window, event, readings, k_fog[E7_NODE], k_store, pub_key, priv_key, rng, n, delegation_active
                )
            else:
                raise ValueError(f"Unknown E7 method: {method}")
            rows.append(row)
    summary = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        totals = [float(row["total_ms"]) for row in method_rows]
        storage = [float(row["storage_bytes_per_window"]) for row in method_rows]
        storage_items = [float(row["storage_items_per_window"]) for row in method_rows]
        fog_latencies = [float(row["fog_latency_ms"]) for row in method_rows]
        summary.append(
            {
                "method": method,
                "windows": len(method_rows),
                "total_ms_median": statistics.median(totals),
                "total_ms_std": statistics.stdev(totals) if len(totals) > 1 else 0.0,
                "fog_latency_ms_median": statistics.median(fog_latencies),
                "storage_items_per_window_median": statistics.median(storage_items),
                "storage_bytes_per_window_median": statistics.median(storage),
                "within_500ms_windows": sum(1 for row in method_rows if row["within_500ms"]),
                "violates_500ms_windows": sum(1 for row in method_rows if not row["within_500ms"]),
            }
        )
    return rows, summary


def run_e8(progress: Callable[[str], None] | None = None) -> list[dict[str, object]]:
    total_sensors = sum(node["sensors"] for node in FOG_NODES.values())
    group_size = FOG_NODES["F1"]["sensors"]
    rows = []
    for scenario in E8_SCENARIOS:
        if progress:
            progress(f"E8 scenario={scenario}")
        if scenario == "one_fog_compromised":
            global_exposed = total_sensors
            scoped_exposed = group_size
            note = "fog-scoped key bounds exposure to one fog group"
        elif scenario == "backup_during_delegation":
            global_exposed = total_sensors
            scoped_exposed = group_size * 2
            note = "backup has its local group plus one delegated source group"
        elif scenario == "kmm_compromised":
            global_exposed = total_sensors
            scoped_exposed = total_sensors
            note = "KMM is the trust anchor for both schemes"
        elif scenario == "host_os_reads_enclave":
            global_exposed = 0
            scoped_exposed = 0
            note = "under simulated SGX assumption, enclave memory blocks host reads"
        else:
            raise ValueError(f"Unknown E8 scenario: {scenario}")
        rows.append(
            {
                "scenario": scenario,
                "total_sensors": total_sensors,
                "global_key_exposed": global_exposed,
                "global_key_exposed_pct": global_exposed / total_sensors * 100.0,
                "fog_scoped_exposed": scoped_exposed,
                "fog_scoped_exposed_pct": scoped_exposed / total_sensors * 100.0,
                "exposure_reduction_pct": (global_exposed - scoped_exposed) / total_sensors * 100.0,
                "model_note": "analytical exposure accounting; not a cryptographic attack simulation",
                "interpretation": note,
            }
        )
    return rows
