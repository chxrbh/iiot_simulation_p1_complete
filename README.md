# IIoT Simulation 

A simulation framework for evaluating privacy-preserving data aggregation in **Industrial IoT (IIoT)** fog-computing environments. The project models Paillier homomorphic encryption, AES-GCM sensor encryption, SGX/TEE enclave processing, and a Key Management Module (KMM) across a network of heterogeneous fog nodes.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Fog Node Configuration](#fog-node-configuration)
- [Results](#results)
- [Notes & Limitations](#notes--limitations)

---

## Overview

This codebase reproduces and repairs a three-part experiment series (P1, P2, P3) studying the trade-offs between cryptographic security and real-time performance in IIoT systems. Key topics include:

- **Storage overhead** of Paillier vs. AES-GCM ciphertexts (E1)
- **Aggregation latency** of 2048-bit Paillier vs. a 500 ms timing window (E2)
- **KMM correctness** under single and multi-source delegation (E3a, E3b)
- **KMM combine overhead** at varying delegation counts (E4)
- **Dynamic task scheduling** with capacity scoring across fog nodes (E5)
- **Fault detection** via ACK/KMM and checkpoint/replication strategies (E6)
- **End-to-end pipeline latency** across the full proposed system (E7)
- **Blast radius analysis** for key compromise scenarios (E8)

> **Security note:** SGX/TEE behaviour is *simulated*. Hardware isolation is a formal assumption, not a live enclave implementation.

---

## Project Structure

```
iiot_simulation_p1_complete-main/
│
├── config.py              # All shared constants and experiment parameters
├── crypto_sim.py          # Cryptographic primitives (Paillier, AES-GCM, KMM, SGX sim)
├── experiments_p1.py      # Runners for E1, E2, E3a, E5
├── experiments_p2.py      # Runners for E3b, E4, E6
├── experiments_p3.py      # Runners for E7, E8
├── figures.py             # Matplotlib figure generation for all experiments
├── results.py             # CSV/summary writing utilities
├── tests.py               # Unit tests
│
├── run_all_p1.py          # Entrypoint: reproduce P1 experiments (E1, E2, E3a, E5)
├── run_all_p2.py          # Entrypoint: reproduce P2 experiments (E3b, E4, E6)
├── run_all_p3.py          # Entrypoint: reproduce P3 experiments (E7, E8)
│
├── requirements.txt       # Python dependencies
│
├── iiot_simulation_p1_complete.ipynb  # Original Jupyter notebook (reference)
│
└── results/               # Auto-generated outputs (CSV, PNG, PDF)
    ├── e1_storage.*
    ├── e2_latency.*
    ├── e3a_correctness.*
    ├── e3b_multisource_correctness.*
    ├── e4_kmm_combine.*
    ├── e5_capacity_score.*
    ├── e6_fault_detection.*
    ├── e7_pipeline_latency.*
    ├── e8_blast_radius.*
    ├── summary.md
    ├── summary_p2.md
    └── summary_p3.md
```

---

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`:

```
cryptography==41.0.7
matplotlib==3.6.3
numpy==1.26.4
pandas==2.2.2
phe==1.5.0
```

> The `phe` library provides the preferred Paillier backend. A fallback pure-Python implementation is included for environments where `phe` is unavailable.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/iiot_simulation_p1_complete.git
cd iiot_simulation_p1_complete

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

Each part of the experiment series has its own entrypoint script.

### Run P1 experiments (E1, E2, E3a, E5)

```bash
python run_all_p1.py
```

### Run P2 experiments (E3b, E4, E6)

```bash
python run_all_p2.py
```

### Run P3 experiments (E7, E8)

```bash
python run_all_p3.py
```

### Common CLI options

All three scripts share the following flags:

| Flag | Default | Description |
|---|---|---|
| `--seed` | `42` | Random seed for reproducibility |
| `--key-bits` | `2048` | Paillier key size in bits |
| `--results-dir` | `results/` | Output directory for CSV and figures |
| `--quick` | *(off)* | Run a fast smoke test (uses 256-bit keys) |

**Example — quick smoke test:**

```bash
python run_all_p1.py --quick
```

**Example — custom seed and output directory:**

```bash
python run_all_p1.py --seed 99 --results-dir my_results/
```

### Run tests

```bash
python tests.py
```

---

## Experiments

| ID | Part | Description |
|---|---|---|
| E1 | P1 | Ciphertext storage comparison: Paillier vs. AES-GCM vs. plaintext |
| E2 | P1 | Aggregation latency of 2048-bit Paillier vs. 500 ms window |
| E3a | P1 | KMM single-source key delegation correctness |
| E3b | P2 | KMM multi-source key delegation correctness |
| E4 | P2 | KMM combine overhead at varying delegation counts (k) |
| E5 | P1 | Dynamic task scheduling with capacity scoring |
| E6 | P2 | Fault detection — ACK/KMM, checkpoint, replication strategies |
| E7 | P3 | End-to-end pipeline latency across all pipeline stages |
| E8 | P3 | Blast radius analysis for key compromise scenarios |

### Key findings

- **E2:** 2048-bit Paillier violates the 500 ms aggregation window before n = 1000 sensors.
- **E7:** The full proposed pipeline also violates the 500 ms window for at least one tested configuration.
- **E8:** Blast radius analysis is analytical — it models exposure scope, not live cryptographic attacks.

---

## Fog Node Configuration

Five heterogeneous fog nodes are modelled (`config.py`):

| Node | Class | CPU Cores | RAM | Bandwidth | Sensors |
|---|---|---|---|---|---|
| F1 | Strong | 4 | 8 GB | 100 Mbps | 20 |
| F2 | Medium | 2 | 4 GB | 50 Mbps | 20 |
| F3 | Weak | 1 | 2 GB | 20 Mbps | 20 |
| F4 | Medium-Fast | 2 | 4 GB | 80 Mbps | 20 |
| F5 | Strong | 4 | 8 GB | 100 Mbps | 20 |

Task scheduling uses a **capacity score** that weights workload, latency, and queue depth differently depending on task type (`LS` vs. compute-heavy).

---

## Results

After running the scripts, outputs are saved to `results/` (or your custom `--results-dir`):

- **CSV files** — raw numerical data for each experiment
- **PNG/PDF figures** — publication-quality plots
- **summary.md / summary_p2.md / summary_p3.md** — human-readable experiment conclusions

---

## Notes & Limitations

- SGX/TEE isolation is **simulated**, not hardware-enforced. All enclave behaviour is a formal assumption.
- Fault detection in E6 uses a **deterministic analytical timing model**, not live distributed fault injection.
- E8 blast radius is **exposure accounting**, not a cryptographic attack simulation or SGX penetration test.
- Results may vary slightly across machines due to Python timing precision, but the `--seed` flag ensures experiment-level reproducibility.
