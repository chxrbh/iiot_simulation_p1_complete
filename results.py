"""CSV and summary helpers for repaired P1 experiments."""

from __future__ import annotations

import csv
import os
import platform
import sys
from importlib import metadata

from config import RESULTS_DIR
from crypto_sim import cpu_label


def ensure_results_dir(results_dir: str = RESULTS_DIR) -> None:
    os.makedirs(results_dir, exist_ok=True)


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def metadata_rows(seed: int, key_bits: int, quick: bool = False, paillier_backend: str = "unknown") -> list[dict[str, object]]:
    return [
        {"key": "seed", "value": seed},
        {"key": "key_bits", "value": key_bits},
        {"key": "paillier_backend", "value": paillier_backend},
        {"key": "quick_smoke_run", "value": quick},
        {"key": "python", "value": sys.version.replace("\n", " ")},
        {"key": "platform", "value": platform.platform()},
        {"key": "cpu", "value": cpu_label()},
        {"key": "cryptography", "value": package_version("cryptography")},
        {"key": "matplotlib", "value": package_version("matplotlib")},
        {"key": "numpy", "value": package_version("numpy")},
        {"key": "pandas", "value": package_version("pandas")},
        {"key": "phe", "value": package_version("phe")},
    ]


def write_summary(
    path: str,
    e2_rows: list[dict[str, object]],
    key_bits: int,
    quick: bool = False,
    paillier_backend: str = "unknown",
) -> None:
    violations = [row for row in e2_rows if not row["ours_within_500ms"]]
    conclusion = (
        "2048-bit Paillier violates the 500 ms aggregation window for at least one tested n."
        if violations and key_bits == 2048
        else "This quick smoke run does not establish 2048-bit timing feasibility."
        if quick
        else "All tested n values are within the 500 ms aggregation window."
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Repaired P1 Experiment Summary\n\n")
        fh.write("This summary is generated from script-produced CSV files, not notebook state.\n\n")
        fh.write(
            f"Run mode: {'quick smoke test' if quick else 'full configured experiment'}; "
            f"key_bits={key_bits}; Paillier backend={paillier_backend}.\n\n"
        )
        fh.write(f"## E2 Latency Conclusion\n{conclusion}\n\n")
        fh.write("Security note: SGX/TEE behavior is simulated; hardware isolation is a formal assumption.\n")
