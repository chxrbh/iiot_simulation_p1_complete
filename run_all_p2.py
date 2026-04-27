"""Single entrypoint for repaired P2 experiment reproduction."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

from config import DEFAULT_KEY_BITS, DEFAULT_SEED, E4_K_VALUES, E4_REPS, E6_FAILURE_SCENARIOS, E6_METHODS, RESULTS_DIR
from crypto_sim import generate_fog_keys, generate_paillier_keypair, paillier_backend_name
from experiments_p2 import run_e3b, run_e4, run_e6
from figures import generate_p2_all
from results import ensure_results_dir, metadata_rows, write_csv, write_p2_summary


class ProgressBar:
    def __init__(self, total: int, label: str, enabled: bool = True) -> None:
        self.total = max(total, 1)
        self.label = label
        self.enabled = enabled
        self.current = 0
        self.started = time.perf_counter()
        self.last_message = ""

    def step(self, message: str = "") -> None:
        self.current += 1
        self.last_message = message
        self.render()

    def render(self, final: bool = False) -> None:
        if not self.enabled:
            return
        done = self.total if final else min(self.current, self.total)
        width = 32
        filled = int(width * done / self.total)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.perf_counter() - self.started
        suffix = f" | {self.last_message}" if self.last_message else ""
        sys.stdout.write(f"\r{self.label} [{bar}] {done}/{self.total} {done / self.total * 100:5.1f}% {elapsed:6.1f}s{suffix}")
        sys.stdout.flush()
        if final or done >= self.total:
            sys.stdout.write("\n")

    def finish(self) -> None:
        if self.current >= self.total:
            return
        self.current = self.total
        self.render(final=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repaired P2 IIoT experiments")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--key-bits", type=int, default=DEFAULT_KEY_BITS)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--quick", action="store_true", help="Smoke-test mode with smaller key and reduced loops.")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = args.seed
    key_bits = 256 if args.quick else args.key_bits
    ensure_results_dir(args.results_dir)
    rng = random.Random(seed)
    print(f"Generating Paillier keypair with key_bits={key_bits}...")
    pub_key, priv_key = generate_paillier_keypair(key_bits, rng)
    backend = paillier_backend_name(pub_key)
    k_fog, k_store = generate_fog_keys(["F1", "F2", "F3", "F4", "F5"], rng)
    show_progress = not args.no_progress
    print(f"Running repaired P2 experiments with seed={seed}, key_bits={key_bits}, backend={backend}")

    e3b_trials = 2 if args.quick else 100
    e3b_progress = ProgressBar(e3b_trials, "E3b multi-source", enabled=show_progress)
    e3b = run_e3b(pub_key, priv_key, k_fog, k_store, seed, trials=e3b_trials, progress=e3b_progress.step)
    e3b_progress.finish()
    write_csv(os.path.join(args.results_dir, "e3b_multisource_correctness.csv"), e3b)

    e4_reps = 1 if args.quick else E4_REPS
    e4_k_values = [2, 5] if args.quick else E4_K_VALUES
    e4_progress = ProgressBar(len(e4_k_values) * e4_reps, "E4 KMM combine", enabled=show_progress)
    e4 = run_e4(pub_key, seed, reps=e4_reps, k_values=e4_k_values, progress=e4_progress.step)
    e4_progress.finish()
    write_csv(os.path.join(args.results_dir, "e4_kmm_combine.csv"), e4)

    e6_progress = ProgressBar(len(E6_FAILURE_SCENARIOS) * len(E6_METHODS), "E6 fault model", enabled=show_progress)
    e6 = run_e6(progress=e6_progress.step)
    e6_progress.finish()
    write_csv(os.path.join(args.results_dir, "e6_fault_detection.csv"), e6)

    write_csv(os.path.join(args.results_dir, "metadata_p2.csv"), metadata_rows(seed, key_bits, quick=args.quick, paillier_backend=backend))
    write_p2_summary(os.path.join(args.results_dir, "summary_p2.md"), key_bits=key_bits, quick=args.quick, paillier_backend=backend)
    generate_p2_all(args.results_dir)
    print(f"Done. P2 results written to {args.results_dir}/")


if __name__ == "__main__":
    main()
