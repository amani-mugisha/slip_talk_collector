from __future__ import annotations

from discovery.build_datasets.Document import Document
import struct 
import hashlib

class Deduplicator:

    def __init__(
        self,
        num_hashes: int = 128,
        num_bands: int = 16,
        similarity_threshold: float = 0.80,
        shingle_size: int = 5,
    ):
        assert num_hashes % num_bands == 0, "num_hashes must be divisible by num_bands"
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.similarity_threshold = similarity_threshold
        self.shingle_size = shingle_size

        # Pre-compute hash parameters (universal hash family)
        _MAX_HASH = (1 << 32) - 1
        rng_state = 42
        self._hash_params: list[tuple[int, int]] = []
        for _ in range(num_hashes):
            rng_state = (rng_state * 1664525 + 1013904223) & 0xFFFFFFFF
            a = rng_state | 1
            rng_state = (rng_state * 1664525 + 1013904223) & 0xFFFFFFFF
            b = rng_state
            self._hash_params.append((a, b))
        self._MAX_HASH = _MAX_HASH

        self._exact: set[str] = set()
        self._band_buckets: list[dict[bytes, list[str]]] = [
            {} for _ in range(num_bands)
        ]
        self._seen_ids: set[str] = set()

    # ── MinHash ─────────────────────────────────────────────────────

    def _shingles(self, text: str) -> set[int]:
        text = text.lower()
        k = self.shingle_size
        return {
            int(hashlib.md5(text[i:i+k].encode()).hexdigest()[:8], 16)
            for i in range(max(1, len(text) - k + 1))
        }

    def minhash(self, text: str) -> list[int]:
        shingles = self._shingles(text)
        if not shingles:
            return [self._MAX_HASH] * self.num_hashes
        signature = []
        for a, b in self._hash_params:
            min_val = min((a * s + b) & self._MAX_HASH for s in shingles)
            signature.append(min_val)
        return signature

    def _band_keys(self, sig: list[int]) -> list[bytes]:
        keys = []
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            band = sig[start:start + self.rows_per_band]
            keys.append(struct.pack(f"{self.rows_per_band}I", *band))
        return keys

    # ── Public API ───────────────────────────────────────────────────

    def is_duplicate(self, doc: Document) -> bool:
        """Return True if document is a near-duplicate of a seen document."""
        # 1. Exact fingerprint
        fp = hashlib.sha256(doc.text.encode()).hexdigest()
        if fp in self._exact:
            doc.is_duplicate = True
            return True
        self._exact.add(fp)

        # 2. MinHash LSH
        sig = self.minhash(doc.text)
        doc.minhash = sig
        band_keys = self._band_keys(sig)

        is_dup = False
        for band_idx, key in enumerate(band_keys):
            bucket = self._band_buckets[band_idx]
            if key in bucket:
                is_dup = True
                break

        if is_dup:
            doc.is_duplicate = True
            return True

        # Register this document
        for band_idx, key in enumerate(band_keys):
            bucket = self._band_buckets[band_idx]
            bucket.setdefault(key, []).append(doc.doc_id)

        doc.is_duplicate = False
        return False