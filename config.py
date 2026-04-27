"""Shared configuration for the repaired P1 IIoT experiments."""

from __future__ import annotations

DEFAULT_SEED = 42
DEFAULT_KEY_BITS = 2048
WINDOW_MS = 500.0
SCALE = 1000
TAU = 0.8
KMM_PROV_MS = 50.0
T_ACK_MS = 100.0
TLS_MS = 20.0
ATTEST_MS = 50.0

RESULTS_DIR = "results"
SEEDS = [42, 43, 44, 45, 46]

E1_N_VALUES = [10, 50, 100, 200, 500, 1000]

# E2 intentionally stops at n=500. The repaired conclusion states that
# 2048-bit Paillier already violates the 500 ms window well before n=1000.
E2_N_VALUES = [10, 50, 100, 200, 500]
E2_REPS = 3

E3_N = 100
E3_TRIALS = 100
E3_K_VALUES = [1, 5, 10, 20, 50]

E5_WINDOWS = 20
E5_TASKS_PER_WINDOW = 100
E5_TO_RATIO = 0.70

E3B_SOURCE_N = 30
E3B_SOURCES = ["F1", "F2"]
E3B_BACKUP = "F4"
E3B_BACKUP_OWN_N = 30
E3B_K_DELEGATED_PER_SOURCE = 5
E3B_TRIALS = 100

E4_K_VALUES = [1, 2, 5, 10, 20, 50, 100]
E4_REPS = 30

E6_FAILURE_SCENARIOS = {
    "fail_0ms": 0.0,
    "mid_window_250ms": 250.0,
    "late_window_450ms": 450.0,
}
E6_METHODS = [
    "b1_gossip",
    "b2_replication",
    "checkpoint",
    "b4_multilayer",
    "b5_fog_clustering",
    "proposed_ack_kmm",
]
E6_SEEDS = list(range(30))
E6_CHECKPOINT_INTERVAL_MS = 500.0
E6_CHECKPOINT_RESTORE_MS = 500.0
E6_GOSSIP_DETECT_MS = 1500.0
E6_MULTILAYER_DETECTION_MS = 500.0
E6_CLUSTER_DETECT_MS = 500.0
E6_CLUSTER_SELECTION_MS = 100.0
E6_CLUSTER_REROUTE_MS = 150.0
E6_ACK_KEY_BYTES = 28

E7_N = 100
E7_NODE = "F2"
E7_NET_SENSOR_TO_FOG_MS = 2.0
E7_NET_SENSOR_TO_CLOUD_MS = 30.0
E7_SENSOR_AES_MS = 0.001
E7_PLAINTEXT_SUM_MS = 1.0
E7_PAILLIER_ENC_MS = 3.8
E7_PAILLIER_ADD_MS = 0.01
E7_KMM_COMBINE_MS = 1.0
E7_STORAGE_PREP_MS = 5.0
E7_CLOUD_UPLOAD_MS = 10.0
E7_TEE_DELEGATION_MS = 150.0
E7_METHODS = ["cloud_only", "fog_plaintext", "paillier_fog_convert", "ours"]
E7_STAGE_COLUMNS = [
    "sensor_to_fog_ms",
    "sensor_to_cloud_ms",
    "sensor_aes_ms",
    "plaintext_sum_ms",
    "enclave_aes_to_paillier_ms",
    "paillier_accum_ms",
    "delegation_ms",
    "kmm_combine_ms",
    "storage_prep_ms",
    "cloud_upload_ms",
]

E8_SCENARIOS = [
    "one_fog_compromised",
    "backup_during_delegation",
    "kmm_compromised",
    "host_os_reads_enclave",
]

FOG_NODES = {
    "F1": {
        "cpu_cap": 4,
        "ram_gb": 8,
        "bw_mbps": 100,
        "latency": 0.2,
        "sensors": 20,
        "class": "Strong",
        "speed_factor": 1.0,
    },
    "F2": {
        "cpu_cap": 2,
        "ram_gb": 4,
        "bw_mbps": 50,
        "latency": 0.4,
        "sensors": 20,
        "class": "Medium",
        "speed_factor": 1.8,
    },
    "F3": {
        "cpu_cap": 1,
        "ram_gb": 2,
        "bw_mbps": 20,
        "latency": 0.8,
        "sensors": 20,
        "class": "Weak",
        "speed_factor": 3.2,
    },
    "F4": {
        "cpu_cap": 2,
        "ram_gb": 4,
        "bw_mbps": 80,
        "latency": 0.2,
        "sensors": 20,
        "class": "Medium-Fast",
        "speed_factor": 1.4,
    },
    "F5": {
        "cpu_cap": 4,
        "ram_gb": 8,
        "bw_mbps": 100,
        "latency": 0.3,
        "sensors": 20,
        "class": "Strong",
        "speed_factor": 1.0,
    },
}

BASE_WORKLOAD = {
    "F1": {"workload": 0.20, "latency": 0.30, "queue": 0.20},
    "F2": {"workload": 0.50, "latency": 0.40, "queue": 0.40},
    "F3": {"workload": 0.60, "latency": 0.80, "queue": 0.55},
    "F4": {"workload": 0.40, "latency": 0.10, "queue": 0.30},
    "F5": {"workload": 0.45, "latency": 0.30, "queue": 0.40},
}


def build_schedule() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for window in range(1, E5_WINDOWS + 1):
        if window <= 5:
            f2_load, f5_alive, event = 0.50, True, "Normal"
        elif window <= 10:
            f2_load, f5_alive, event = 0.85, True, "F2 overload"
        elif window == 11:
            f2_load, f5_alive, event = 0.50, False, "F5 fails"
        elif window <= 13:
            f2_load, f5_alive, event = 0.50, False, "Gossip detects F5"
        elif window <= 17:
            f2_load, f5_alive, event = 0.50, True, "Recovery"
        else:
            f2_load, f5_alive, event = 0.90, True, "F2 second overload"
        rows.append(
            {
                "window": window,
                "f2_load": f2_load,
                "f5_alive": f5_alive,
                "event": event,
            }
        )
    return rows


def capacity_score(node_id: str, workload_table: dict[str, dict[str, float]], task_type: str) -> float:
    node = workload_table[node_id]
    if task_type == "LS":
        w1, w2, w3 = 0.20, 0.50, 0.30
    else:
        w1, w2, w3 = 0.50, 0.15, 0.35
    return w1 * (1 - node["workload"]) + w2 * (1 - node["latency"]) + w3 * (1 - node["queue"])
