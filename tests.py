"""Unit and experiment-shape tests for the repaired P1 package."""

from __future__ import annotations

import random
import unittest

from config import (
    E2_N_VALUES,
    E3_K_VALUES,
    E3_N,
    E3_TRIALS,
    E3B_TRIALS,
    E4_K_VALUES,
    E6_FAILURE_SCENARIOS,
    E6_METHODS,
    E7_METHODS,
    E7_STAGE_COLUMNS,
    E8_SCENARIOS,
    KMM_PROV_MS,
    SCALE,
    T_ACK_MS,
    capacity_score,
)
from crypto_sim import (
    KMM,
    aes_decrypt,
    aes_encrypt,
    generate_fog_keys,
    generate_paillier_keypair,
    paillier_encrypt,
    sgx_enclave_process,
)
from experiments_p1 import run_e1, run_e2, run_e3a, run_e5
from experiments_p2 import run_e3b, run_e4, run_e6
from experiments_p3 import run_e7, run_e8


class P1RepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = random.Random(7)
        self.pub, self.priv = generate_paillier_keypair(256, self.rng)
        self.k_fog, self.k_store = generate_fog_keys(["F1", "F2", "F3", "F4", "F5"], self.rng)

    def test_aes_round_trip(self) -> None:
        nonce, ct = aes_encrypt(self.k_fog["F1"], 12.345, self.rng)
        self.assertEqual(aes_decrypt(self.k_fog["F1"], nonce, ct), 12.345)

    def test_paillier_scaled_addition(self) -> None:
        a = paillier_encrypt(self.pub, 12 * SCALE, self.rng)
        b = paillier_encrypt(self.pub, 7 * SCALE, self.rng)
        self.assertEqual(self.priv.decrypt(a + b), 19 * SCALE)

    def test_kmm_combine(self) -> None:
        kmm = KMM(self.k_fog)
        c1 = paillier_encrypt(self.pub, 4, self.rng)
        c2 = paillier_encrypt(self.pub, 9, self.rng)
        combined, _ = kmm.combine({"F1": c1, "F4": c2}, self.pub)
        self.assertEqual(self.priv.decrypt(combined), 13)

    def test_key_provision_revoke(self) -> None:
        kmm = KMM(self.k_fog)
        key, latency = kmm.provision_key("F1", "F4")
        self.assertEqual(key, self.k_fog["F1"])
        self.assertEqual(latency, 50.0)
        self.assertIn("F1", kmm.delegated["F4"])
        kmm.revoke_key("F1", "F4")
        self.assertNotIn("F1", kmm.delegated["F4"])

    def test_capacity_score_targets(self) -> None:
        wt = {
            "F1": {"workload": 0.20, "latency": 0.30, "queue": 0.20},
            "F3": {"workload": 0.60, "latency": 0.80, "queue": 0.55},
            "F4": {"workload": 0.40, "latency": 0.10, "queue": 0.30},
        }
        self.assertEqual(max(wt, key=lambda nid: capacity_score(nid, wt, "LS")), "F4")
        self.assertEqual(max(wt, key=lambda nid: capacity_score(nid, wt, "TO")), "F1")

    def test_e1_formulas(self) -> None:
        rows = run_e1(self.pub)
        row100 = next(row for row in rows if row["n"] == 100)
        self.assertEqual(row100["ours_ciphertexts"], 1)
        self.assertEqual(row100["ciphertext_count_reduction"], 100)
        self.assertAlmostEqual(row100["byte_reduction_vs_aes"], row100["aes_bytes"] / row100["ours_cloud_aes_bytes"])

    def test_e2_shape_quick(self) -> None:
        rows = run_e2(self.pub, self.priv, self.k_fog, self.k_store, seed=7, reps=1, n_values=[10])
        self.assertEqual([row["n"] for row in rows], [10])
        for key in ["plaintext_ms_median", "aes_ms_median", "paillier_nobatch_ms_median", "ours_ms_median"]:
            self.assertIn(key, rows[0])

    def test_e3a_shape_quick_and_config(self) -> None:
        self.assertEqual(E3_N, 100)
        self.assertEqual(E3_TRIALS, 100)
        self.assertEqual(E3_K_VALUES, [1, 5, 10, 20, 50])
        rows = run_e3a(self.pub, self.priv, self.k_fog, self.k_store, seed=7, trials=2, n=10, k_values=[1])
        self.assertEqual(rows[0]["accuracy_scaled_pct"], 100.0)
        self.assertIn("median_quantization_error", rows[0])

    def test_e5_aggregate(self) -> None:
        runs, aggregate = run_e5(seeds=[1, 2])
        self.assertEqual(len(runs), 8)
        self.assertEqual({row["method"] for row in aggregate}, {"random", "round_robin", "threshold", "capacity"})
        for row in aggregate:
            self.assertIn("deadline_satisfaction_pct_ci95", row)

    def test_e3b_multisource_shape_and_edges(self) -> None:
        rows = run_e3b(self.pub, self.priv, self.k_fog, self.k_store, seed=7, trials=2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(E3B_TRIALS, 100)
        self.assertEqual(rows[0]["accuracy_scaled_pct"], 100.0)
        self.assertEqual(rows[0]["provisioned_edges"], "F1->F4;F2->F4")

    def test_e4_kmm_combine_shape_and_bytes(self) -> None:
        rows = run_e4(self.pub, seed=7, reps=1, k_values=[1, 2, 5])
        self.assertEqual([row["k_fog_aggregates"] for row in rows], [1, 2, 5])
        for row in rows:
            self.assertEqual(row["he_additions"], row["k_fog_aggregates"] - 1)
            self.assertEqual(row["bytes_received"], row["k_fog_aggregates"] * row["ciphertext_bytes"])
        self.assertEqual(E4_K_VALUES, [1, 2, 5, 10, 20, 50, 100])

    def test_e6_fault_model_pairs_and_ack_bound(self) -> None:
        rows = run_e6()
        self.assertEqual(len(rows), len(E6_FAILURE_SCENARIOS) * len(E6_METHODS))
        ack_rows = [row for row in rows if row["method"] == "ack_kmm"]
        for row in ack_rows:
            remaining = row["window_ms"] - row["failure_time_ms"]
            self.assertEqual(row["data_loss_ms"], min(remaining, T_ACK_MS + KMM_PROV_MS))
        before = [row for row in rows if row["scenario"] == "before_window"]
        none = next(row for row in before if row["method"] == "none")
        replication = next(row for row in before if row["method"] == "replication")
        ack = next(row for row in before if row["method"] == "ack_kmm")
        self.assertGreater(replication["storage_overhead_factor"], ack["storage_overhead_factor"])
        self.assertGreaterEqual(none["data_loss_ms"], ack["data_loss_ms"])

    def test_e7_pipeline_shape_totals_and_storage(self) -> None:
        rows, summary = run_e7(
            self.pub,
            self.priv,
            self.k_fog,
            self.k_store,
            seed=7,
            n=5,
            windows=2,
        )
        self.assertEqual(len(rows), 2 * len(E7_METHODS))
        self.assertEqual({row["method"] for row in rows}, set(E7_METHODS))
        self.assertEqual({row["method"] for row in summary}, set(E7_METHODS))
        for row in rows:
            for column in E7_STAGE_COLUMNS:
                self.assertIn(column, row)
            stage_total = sum(float(row[column]) for column in E7_STAGE_COLUMNS)
            self.assertAlmostEqual(float(row["total_ms"]), stage_total, places=9)
        storage_by_method = {row["method"]: row["storage_bytes_per_window"] for row in rows if row["window"] == 1}
        self.assertGreater(storage_by_method["cloud_only"], storage_by_method["fog_plaintext"])
        self.assertGreater(storage_by_method["paillier_nobatch"], storage_by_method["ours"])
        self.assertNotEqual(storage_by_method["fog_plaintext"], storage_by_method["ours"])

    def test_e8_blast_radius(self) -> None:
        rows = run_e8()
        self.assertEqual([row["scenario"] for row in rows], E8_SCENARIOS)
        by_scenario = {row["scenario"]: row for row in rows}
        one_fog = by_scenario["one_fog_compromised"]
        self.assertEqual(one_fog["global_key_exposed_pct"], 100.0)
        self.assertEqual(one_fog["fog_scoped_exposed_pct"], 20.0)
        backup = by_scenario["backup_during_delegation"]
        self.assertEqual(backup["fog_scoped_exposed_pct"], 40.0)
        kmm = by_scenario["kmm_compromised"]
        self.assertEqual(kmm["global_key_exposed_pct"], 100.0)
        self.assertEqual(kmm["fog_scoped_exposed_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
