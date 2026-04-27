"""Unit and experiment-shape tests for the repaired P1 package."""

from __future__ import annotations

import random
import unittest

from config import E2_N_VALUES, E3_K_VALUES, E3_N, E3_TRIALS, SCALE, capacity_score
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


if __name__ == "__main__":
    unittest.main()
