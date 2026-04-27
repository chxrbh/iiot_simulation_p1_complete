"""Cryptographic and KMM simulation primitives for P1 experiments.

The Paillier implementation is intentionally small so the experiments remain
reproducible without a notebook-only dependency. It is suitable for simulation,
not for production cryptographic deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
import os
import random
import time

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from config import KMM_PROV_MS, SCALE


def aes_key(rng: random.Random) -> bytes:
    return rng.randbytes(16)


def aes_encrypt(key: bytes, value: float, rng: random.Random) -> tuple[bytes, bytes]:
    nonce = rng.randbytes(12)
    ct = AESGCM(key).encrypt(nonce, str(value).encode("ascii"), None)
    return nonce, ct


def aes_decrypt(key: bytes, nonce: bytes, ct: bytes) -> float:
    return float(AESGCM(key).decrypt(nonce, ct, None).decode("ascii"))


def _lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b


def _is_probable_prime(n: int, rng: random.Random, rounds: int = 12) -> bool:
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2
    for _ in range(rounds):
        a = rng.randrange(2, n - 2)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _prime(bits: int, rng: random.Random) -> int:
    while True:
        candidate = rng.getrandbits(bits) | (1 << (bits - 1)) | 1
        if _is_probable_prime(candidate, rng):
            return candidate


@dataclass(frozen=True)
class PaillierPublicKey:
    n: int

    @property
    def n_square(self) -> int:
        return self.n * self.n

    def encrypt(self, plaintext: int, rng: random.Random | None = None) -> "EncryptedNumber":
        rng = rng or random.SystemRandom()
        n_sq = self.n_square
        while True:
            r = rng.randrange(1, self.n)
            if gcd(r, self.n) == 1:
                break
        # With g=n+1, g^m mod n^2 is 1+n*m for m < n.
        c = ((1 + self.n * (plaintext % self.n)) * pow(r, self.n, n_sq)) % n_sq
        return EncryptedNumber(self, c)


@dataclass(frozen=True)
class PaillierPrivateKey:
    public_key: PaillierPublicKey
    lambda_value: int
    mu: int

    def decrypt(self, encrypted: "EncryptedNumber") -> int:
        n = self.public_key.n
        x = pow(encrypted.ciphertext, self.lambda_value, n * n)
        l_value = (x - 1) // n
        return (l_value * self.mu) % n


@dataclass(frozen=True)
class EncryptedNumber:
    public_key: PaillierPublicKey
    ciphertext: int

    def __add__(self, other: "EncryptedNumber") -> "EncryptedNumber":
        if self.public_key.n != other.public_key.n:
            raise ValueError("Cannot add ciphertexts from different Paillier keys")
        return EncryptedNumber(
            self.public_key,
            (self.ciphertext * other.ciphertext) % self.public_key.n_square,
        )

    def byte_size(self) -> int:
        return max(1, (self.ciphertext.bit_length() + 7) // 8)


def generate_paillier_keypair(bits: int, rng: random.Random) -> tuple[PaillierPublicKey, PaillierPrivateKey]:
    half = bits // 2
    p = _prime(half, rng)
    q = _prime(bits - half, rng)
    while p == q:
        q = _prime(bits - half, rng)
    n = p * q
    pub = PaillierPublicKey(n)
    lam = _lcm(p - 1, q - 1)
    x = pow(n + 1, lam, n * n)
    l_value = (x - 1) // n
    mu = pow(l_value, -1, n)
    return pub, PaillierPrivateKey(pub, lam, mu)


def paillier_ciphertext_bytes(pub_key: PaillierPublicKey) -> int:
    return (pub_key.n_square.bit_length() + 7) // 8


def sgx_enclave_process(
    k_fog: bytes,
    nonce: bytes,
    ct: bytes,
    pub_key: PaillierPublicKey,
    rng: random.Random,
) -> EncryptedNumber:
    value = aes_decrypt(k_fog, nonce, ct)
    return pub_key.encrypt(int(value * SCALE), rng)


def sgx_enclave_storage_prep(
    c_agg_final: EncryptedNumber,
    priv_key: PaillierPrivateKey,
    k_store: bytes,
    rng: random.Random,
) -> tuple[bytes, bytes, float]:
    decoded = priv_key.decrypt(c_agg_final) / SCALE
    nonce, ct = aes_encrypt(k_store, decoded, rng)
    return nonce, ct, decoded


class KMM:
    def __init__(self, k_fog_map: dict[str, bytes]):
        self.k_fog_map = k_fog_map
        self.delegated: dict[str, set[str]] = {}
        self.prov_log: list[dict[str, str]] = []

    def provision_key(self, from_node: str, to_node: str) -> tuple[bytes, float]:
        self.delegated.setdefault(to_node, set()).add(from_node)
        self.prov_log.append({"from": from_node, "to": to_node})
        return self.k_fog_map[from_node], KMM_PROV_MS

    def revoke_key(self, from_node: str, to_node: str) -> None:
        if to_node in self.delegated:
            self.delegated[to_node].discard(from_node)

    def combine(self, aggregates: dict[str, EncryptedNumber], pub_key: PaillierPublicKey) -> tuple[EncryptedNumber, float]:
        t0 = time.perf_counter()
        result = pub_key.encrypt(0)
        for ciphertext in aggregates.values():
            result = result + ciphertext
        return result, (time.perf_counter() - t0) * 1000.0


def generate_fog_keys(node_ids: list[str], rng: random.Random) -> tuple[dict[str, bytes], bytes]:
    return {node_id: aes_key(rng) for node_id in node_ids}, aes_key(rng)


def cpu_label() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return os.uname().machine if hasattr(os, "uname") else "unknown"
