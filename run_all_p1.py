"""Single entrypoint for repaired P1 experiment reproduction."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

from config import DEFAULT_KEY_BITS, DEFAULT_SEED, RESULTS_DIR
from crypto_sim import generate_fog_keys, generate_paillier_keypair, paillier_backend_name
from experiments_p1 import run_e1, run_e2, run_e3a, run_e5
from figures import generate_all
from results import ensure_results_dir, metadata_rows, write_csv, write_summary


class ProgressBar:
    def __init__(self, total: int, label: str = "Progress", enabled: bool = True) -> None:
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
        pct = done / self.total * 100
        elapsed = time.perf_counter() - self.started
        suffix = f" | {self.last_message}" if self.last_message else ""
        sys.stdout.write(f"\r{self.label} [{bar}] {done}/{self.total} {pct:5.1f}% {elapsed:6.1f}s{suffix}")
        sys.stdout.flush()
        if final or done >= self.total:
            sys.stdout.write("\n")

    def finish(self) -> None:
        if self.current >= self.total:
            return
        self.current = self.total
        self.render(final=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repaired P1 IIoT experiments")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--key-bits", type=int, default=DEFAULT_KEY_BITS)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test mode: smaller key and reduced E2/E3 loops for local validation.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress bars.",
    )
    parser.add_argument(
        "--show-figures",
        action="store_true",
        help="Display figures with matplotlib after saving them, similar to notebook plt.show().",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick and args.results_dir == RESULTS_DIR:
        args.results_dir = f"{RESULTS_DIR}_quick"
    seed = args.seed
    key_bits = 256 if args.quick else args.key_bits
    ensure_results_dir(args.results_dir)
    rng = random.Random(seed)
    print(f"Generating Paillier keypair with key_bits={key_bits}...")
    pub_key, priv_key = generate_paillier_keypair(key_bits, rng)
    backend = paillier_backend_name(pub_key)
    k_fog, k_store = generate_fog_keys(["F1", "F2", "F3", "F4", "F5"], rng)

    print(f"Running repaired P1 experiments with seed={seed}, key_bits={key_bits}, paillier_backend={backend}")
    show_progress = not args.no_progress
    e1 = run_e1(pub_key)
    write_csv(os.path.join(args.results_dir, "e1_storage.csv"), e1)

    e2_reps = 1 if args.quick else None
    e2_n = [10, 50] if args.quick else None
    e2_total = len(e2_n or [10, 50, 100, 200, 500]) * (e2_reps or 3)
    e2_progress = ProgressBar(e2_total, "E2 latency", enabled=show_progress)
    e2 = run_e2(
        pub_key,
        priv_key,
        k_fog,
        k_store,
        seed,
        reps=e2_reps or 3,
        n_values=e2_n,
        progress=e2_progress.step,
    )
    e2_progress.finish()
    write_csv(os.path.join(args.results_dir, "e2_latency.csv"), e2)

    e3_trials = 2 if args.quick else 100
    e3_n = 10 if args.quick else 100
    e3_k = [1, 5] if args.quick else None
    e3_total = len(e3_k or [1, 5, 10, 20, 50]) * e3_trials
    e3_progress = ProgressBar(e3_total, "E3a correctness", enabled=show_progress)
    e3a = run_e3a(
        pub_key,
        priv_key,
        k_fog,
        k_store,
        seed,
        trials=e3_trials,
        n=e3_n,
        k_values=e3_k,
        progress=e3_progress.step,
    )
    e3_progress.finish()
    write_csv(os.path.join(args.results_dir, "e3a_correctness.csv"), e3a)

    e5_progress = ProgressBar(4 * 5, "E5 task simulation", enabled=show_progress)
    e5_runs, e5_agg = run_e5(progress=e5_progress.step)
    e5_progress.finish()
    write_csv(os.path.join(args.results_dir, "e5_capacity_score_runs.csv"), e5_runs)
    write_csv(os.path.join(args.results_dir, "e5_capacity_score.csv"), e5_agg)
    write_csv(
        os.path.join(args.results_dir, "metadata.csv"),
        metadata_rows(seed, key_bits, quick=args.quick, paillier_backend=backend),
    )
    write_summary(
        os.path.join(args.results_dir, "summary.md"),
        e2,
        key_bits=key_bits,
        quick=args.quick,
        paillier_backend=backend,
    )
    generate_all(args.results_dir, show=args.show_figures)
    print(f"Done. Results written to {args.results_dir}/")


if __name__ == "__main__":
    main()
