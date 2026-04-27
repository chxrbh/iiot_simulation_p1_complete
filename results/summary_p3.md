# Repaired P3 Experiment Summary

This summary is generated from script-produced CSV files, not notebook state.

Run mode: full configured experiment; key_bits=2048; Paillier backend=phe.

## E7 End-to-End Pipeline
The full proposed pipeline violates the 500 ms window for at least one tested window.

E7 note: baseline timing is a deterministic simulation mapped to the reference roles: cloud-only stores n raw values, fog plaintext stores one insecure sum, the Paillier fog-convert baseline stores n ciphertexts, and the proposed method stores one Paillier aggregate. The proposed method is not claimed to reduce Paillier computation time; its E7 claim is privacy-preserving n-to-1 storage reduction, with delegation overhead only in affected windows.

## E8 Blast Radius
E8 is analytical exposure accounting, not cryptographic attack simulation or SGX penetration testing.
