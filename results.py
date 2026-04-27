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


def _metadata_map(rows: list[dict[str, str]]) -> dict[str, str]:
    return {row["key"]: row["value"] for row in rows}


def validate_p2_results(
    results_dir: str,
    *,
    quick: bool,
    key_bits: int,
    expected_ciphertext_bytes: int,
) -> None:
    """Reject mixed quick/full P2 artifacts before figure generation."""
    metadata_path = os.path.join(results_dir, "metadata_p2.csv")
    e3b_path = os.path.join(results_dir, "e3b_multisource_correctness.csv")
    e4_path = os.path.join(results_dir, "e4_kmm_combine.csv")
    for path in [metadata_path, e3b_path, e4_path, os.path.join(results_dir, "e6_fault_detection.csv")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required P2 artifact: {path}")

    metadata_values = _metadata_map(read_csv(metadata_path))
    if metadata_values.get("quick_smoke_run") != str(quick):
        raise ValueError(
            f"P2 metadata quick_smoke_run={metadata_values.get('quick_smoke_run')} "
            f"does not match current run quick={quick}"
        )
    if int(metadata_values.get("key_bits", "0")) != key_bits:
        raise ValueError(
            f"P2 metadata key_bits={metadata_values.get('key_bits')} does not match current run key_bits={key_bits}"
        )

    e3b_rows = read_csv(e3b_path)
    expected_trials = 2 if quick else 100
    e3b_trials = int(float(e3b_rows[0]["trials"]))
    if e3b_trials != expected_trials:
        raise ValueError(f"E3b trials={e3b_trials} does not match expected {expected_trials}")

    e4_rows = read_csv(e4_path)
    for row in e4_rows:
        ciphertext_bytes = int(float(row["ciphertext_bytes"]))
        if ciphertext_bytes != expected_ciphertext_bytes:
            raise ValueError(
                f"E4 ciphertext_bytes={ciphertext_bytes} does not match expected "
                f"{expected_ciphertext_bytes} for key_bits={key_bits}"
            )


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
        fh.write(
            "Interpretation: E2 is a measured Python cryptographic pipeline. For 2048-bit Paillier, "
            "these results do not support a 500 ms real-time feasibility claim.\n\n"
        )
        fh.write(
            "Security note: SGX/TEE behavior is simulated; E3-style correctness checks are not security proofs, "
            "and hardware isolation remains a formal assumption.\n"
        )


def write_p2_summary(
    path: str,
    key_bits: int,
    quick: bool = False,
    paillier_backend: str = "unknown",
) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Repaired P2 Experiment Summary\n\n")
        fh.write("This summary is generated from script-produced CSV files, not notebook state.\n\n")
        fh.write(
            f"Run mode: {'quick smoke test' if quick else 'full configured experiment'}; "
            f"key_bits={key_bits}; Paillier backend={paillier_backend}.\n\n"
        )
        fh.write("Implemented P2 experiments: E3b multi-source correctness, E4 KMM combine overhead, and E6 ACK/KMM fault recovery.\n\n")
        fh.write(
            "E3b note: multi-source correctness is a scaled-sum arithmetic sanity check; it is not a cryptographic "
            "security proof or an SGX isolation experiment.\n\n"
        )
        fh.write(
            "E6 note: fault recovery is an analytical reading-loss model comparing B1 gossip-based detection, "
            "B2 replication-based fault tolerance, B3 checkpoint/restart, B4 multilayer detection, "
            "B5 fog-clustering fault tolerance, and proposed ACK+KMM.\n"
        )


def write_p3_summary(
    path: str,
    e7_summary_rows: list[dict[str, object]],
    key_bits: int,
    quick: bool = False,
    paillier_backend: str = "unknown",
) -> None:
    ours = next((row for row in e7_summary_rows if row["method"] == "ours"), None)
    if quick:
        e7_conclusion = "This quick smoke run does not establish full 2048-bit end-to-end timing feasibility."
    elif ours and int(ours["violates_500ms_windows"]) > 0 and key_bits == 2048:
        e7_conclusion = "The full proposed pipeline violates the 500 ms window for at least one tested window."
    else:
        e7_conclusion = "All tested proposed-pipeline windows are within the 500 ms window."
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Repaired P3 Experiment Summary\n\n")
        fh.write("This summary is generated from script-produced CSV files, not notebook state.\n\n")
        fh.write(
            f"Run mode: {'quick smoke test' if quick else 'full configured experiment'}; "
            f"key_bits={key_bits}; Paillier backend={paillier_backend}.\n\n"
        )
        fh.write(f"## E7 End-to-End Pipeline\n{e7_conclusion}\n\n")
        fh.write(
            "E7 note: baseline timing is a deterministic simulation mapped to the reference roles: "
            "cloud-only stores n raw values, fog plaintext stores one insecure sum, the Paillier fog-convert baseline stores n ciphertexts, "
            "and the proposed method stores one Paillier aggregate. The proposed method is not claimed to reduce Paillier computation time; "
            "its E7 claim is privacy-preserving n-to-1 storage reduction, with delegation overhead only in affected windows.\n\n"
        )
        fh.write("## E8 Blast Radius\nE8 is analytical exposure accounting, not cryptographic attack simulation or SGX penetration testing.\n")
