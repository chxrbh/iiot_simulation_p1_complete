"""Experiment runners for P3 experiments: E7 and E8."""

from __future__ import annotations

from collections.abc import Callable
import random
import statistics
import time

from config import (
    E7_CLOUD_UPLOAD_MS,
    E7_METHODS,
    E7_N,
    E7_NODE,
    E7_STAGE_COLUMNS,
    E8_SCENARIOS,
    FOG_NODES,
    KMM_PROV_MS,
    SCALE,
    TAU,
    WINDOW_MS,
    build_schedule,
)
from crypto_sim import (
    KMM,
    aes_decrypt,
    aes_encrypt,
    paillier_ciphertext_bytes,
    paillier_encrypt,
    sgx_enclave_process,
    sgx_enclave_storage_prep,
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
        "storage_bytes_per_window": 0,
        "privacy_note": "",
    }


def _sensor_encrypt(readings: list[float], key: bytes, rng: random.Random) -> tuple[list[tuple[bytes, bytes]], float]:
    t0 = time.perf_counter()
    pairs = [aes_encrypt(key, value, rng) for value in readings]
    return pairs, (time.perf_counter() - t0) * 1000.0


def _finish_e7_row(row: dict[str, object]) -> dict[str, object]:
    total = sum(float(row[column]) for column in E7_STAGE_COLUMNS)
    row["total_ms"] = total
    row["within_500ms"] = total <= WINDOW_MS
    row["conclusion"] = "within_500ms_window" if total <= WINDOW_MS else "violates_500ms_window"
    return row


def _run_cloud_only(window: int, event: str, readings: list[float], n: int, delegation_active: bool) -> dict[str, object]:
    row = _stage_row_base(window, event, "cloud_only", n, delegation_active)
    row["cloud_upload_ms"] = E7_CLOUD_UPLOAD_MS
    row["storage_bytes_per_window"] = n * 8
    row["privacy_note"] = "no fog privacy; raw readings are stored/transmitted"
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
    pairs, sensor_ms = _sensor_encrypt(readings, key, rng)
    t0 = time.perf_counter()
    _ = sum(aes_decrypt(key, nonce, ct) for nonce, ct in pairs)
    plaintext_ms = (time.perf_counter() - t0) * 1000.0
    row["sensor_aes_ms"] = sensor_ms + plaintext_ms
    row["cloud_upload_ms"] = E7_CLOUD_UPLOAD_MS
    row["storage_bytes_per_window"] = 8
    row["privacy_note"] = "fog decrypts readings; insecure baseline with one aggregate"
    return _finish_e7_row(row)


def _run_paillier_nobatch(
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
    row = _stage_row_base(window, event, "paillier_nobatch", n, delegation_active)
    pairs, sensor_ms = _sensor_encrypt(readings, key, rng)
    ciphertexts = []
    t0 = time.perf_counter()
    for nonce, ct in pairs:
        ciphertexts.append(sgx_enclave_process(key, nonce, ct, pub_key, rng))
    enclave_ms = (time.perf_counter() - t0) * 1000.0
    t_accum = time.perf_counter()
    aggregate = paillier_encrypt(pub_key, 0, rng)
    for ciphertext in ciphertexts:
        aggregate = aggregate + ciphertext
    accum_ms = (time.perf_counter() - t_accum) * 1000.0
    t_store = time.perf_counter()
    _ = sgx_enclave_storage_prep(aggregate, priv_key, store_key, rng)
    storage_prep_ms = (time.perf_counter() - t_store) * 1000.0
    row["sensor_aes_ms"] = sensor_ms
    row["enclave_aes_to_paillier_ms"] = enclave_ms
    row["paillier_accum_ms"] = accum_ms
    row["storage_prep_ms"] = storage_prep_ms
    row["cloud_upload_ms"] = E7_CLOUD_UPLOAD_MS
    row["storage_bytes_per_window"] = n * paillier_ciphertext_bytes(pub_key)
    row["privacy_note"] = "Paillier privacy but no slot aggregation storage benefit"
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
    pairs, sensor_ms = _sensor_encrypt(readings, key, rng)
    kmm = KMM({E7_NODE: key, "F4": key})
    agg_primary = paillier_encrypt(pub_key, 0, rng)
    agg_backup = paillier_encrypt(pub_key, 0, rng)
    delegated = set(rng.sample(range(n), max(1, n // 10))) if delegation_active else set()
    delegated_key = key
    row["delegation_ms"] = 0.0
    if delegation_active:
        delegated_key, row["delegation_ms"] = kmm.provision_key(E7_NODE, "F4")
    enclave_ms = 0.0
    accum_ms = 0.0
    for idx, (nonce, ct) in enumerate(pairs):
        t_enc = time.perf_counter()
        encrypted = sgx_enclave_process(delegated_key if idx in delegated else key, nonce, ct, pub_key, rng)
        enclave_ms += (time.perf_counter() - t_enc) * 1000.0
        t_add = time.perf_counter()
        if idx in delegated:
            agg_primary = agg_primary + paillier_encrypt(pub_key, 0, rng)
            agg_backup = agg_backup + encrypted
        else:
            agg_primary = agg_primary + encrypted
        accum_ms += (time.perf_counter() - t_add) * 1000.0
    combined, kmm_ms = kmm.combine({E7_NODE: agg_primary, "F4": agg_backup}, pub_key)
    t_store = time.perf_counter()
    _ = sgx_enclave_storage_prep(combined, priv_key, store_key, rng)
    storage_prep_ms = (time.perf_counter() - t_store) * 1000.0
    if delegation_active:
        kmm.revoke_key(E7_NODE, "F4")
    row["sensor_aes_ms"] = sensor_ms
    row["enclave_aes_to_paillier_ms"] = enclave_ms
    row["paillier_accum_ms"] = accum_ms
    row["kmm_combine_ms"] = kmm_ms
    row["storage_prep_ms"] = storage_prep_ms
    row["cloud_upload_ms"] = E7_CLOUD_UPLOAD_MS
    row["storage_bytes_per_window"] = 44
    row["privacy_note"] = "slot aggregation with simulated TEE and fog-scoped key provisioning"
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
            elif method == "paillier_nobatch":
                row = _run_paillier_nobatch(
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
        summary.append(
            {
                "method": method,
                "windows": len(method_rows),
                "total_ms_median": statistics.median(totals),
                "total_ms_std": statistics.stdev(totals) if len(totals) > 1 else 0.0,
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
