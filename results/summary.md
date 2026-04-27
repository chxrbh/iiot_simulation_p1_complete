# Repaired P1 Experiment Summary

This summary is generated from script-produced CSV files, not notebook state.

Run mode: full configured experiment; key_bits=2048; Paillier backend=phe.

## E2 Latency Conclusion
2048-bit Paillier violates the 500 ms aggregation window for at least one tested n.

Interpretation: E2 is a measured Python cryptographic pipeline. For 2048-bit Paillier, these results do not support a 500 ms real-time feasibility claim.

Security note: SGX/TEE behavior is simulated; E3-style correctness checks are not security proofs, and hardware isolation remains a formal assumption.
