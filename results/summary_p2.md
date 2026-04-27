# Repaired P2 Experiment Summary

This summary is generated from script-produced CSV files, not notebook state.

Run mode: full configured experiment; key_bits=2048; Paillier backend=phe.

Implemented P2 experiments: E3b multi-source correctness, E4 KMM combine overhead, and E6 ACK/KMM fault recovery.

E6 note: fault recovery is an analytical reading-loss model comparing B1 gossip-based detection, B2 replication-based fault tolerance, B3 checkpoint/restart, B4 multilayer detection, B5 fog-clustering fault tolerance, and proposed ACK+KMM.
