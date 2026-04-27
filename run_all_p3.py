"""Single entrypoint for repaired P3 experiment reproduction."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

from config import DEFAULT_KEY_BITS, DEFAULT_SEED, E7_METHODS, E8_SCENARIOS, RESULTS_DIR
from crypto_sim import generate_fog_keys, generate_paillier_keypair, paillier_backend_name
from experiments_p3 import run_e7, run_e8
from figures import generate_p3_all
from results import ensure_results_dir, metadata_rows, write_csv, write_p3_summary


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
    parser = argparse.ArgumentParser(description="Run repaired P3 IIoT experiments")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--key-bits", type=int, default=DEFAULT_KEY_BITS)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--quick", action="store_true", help="Smoke-test mode with smaller key and reduced loops.")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress bars.")
    parser.add_argument("--show-figures", action="store_true", help="Display figures with matplotlib after saving them.")
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
    show_progress = not args.no_progress
    print(f"Running repaired P3 experiments with seed={seed}, key_bits={key_bits}, backend={backend}")

    e7_windows = 2 if args.quick else None
    e7_n = 10 if args.quick else 100
    e7_total = (e7_windows or 20) * len(E7_METHODS)
    e7_progress = ProgressBar(e7_total, "E7 pipeline", enabled=show_progress)
    e7_rows, e7_summary = run_e7(
        pub_key,
        priv_key,
        k_fog,
        k_store,
        seed,
        n=e7_n,
        windows=e7_windows,
        progress=e7_progress.step,
    )
    e7_progress.finish()
    write_csv(os.path.join(args.results_dir, "e7_pipeline_latency.csv"), e7_rows)
    write_csv(os.path.join(args.results_dir, "e7_pipeline_latency_summary.csv"), e7_summary)

    e8_progress = ProgressBar(len(E8_SCENARIOS), "E8 blast radius", enabled=show_progress)
    e8_rows = run_e8(progress=e8_progress.step)
    e8_progress.finish()
    write_csv(os.path.join(args.results_dir, "e8_blast_radius.csv"), e8_rows)

    write_csv(os.path.join(args.results_dir, "metadata_p3.csv"), metadata_rows(seed, key_bits, quick=args.quick, paillier_backend=backend))
    write_p3_summary(
        os.path.join(args.results_dir, "summary_p3.md"),
        e7_summary,
        key_bits=key_bits,
        quick=args.quick,
        paillier_backend=backend,
    )
    generate_p3_all(args.results_dir, show=args.show_figures)
    print(f"Done. P3 results written to {args.results_dir}/")


if __name__ == "__main__":
    main()
