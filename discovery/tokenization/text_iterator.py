from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from pathlib import Path
from typing import Callable, Generator, Iterator, Optional

import pyarrow.parquet as pq

log = logging.getLogger("text_iterator")


# ── Default text normaliser (applied before yielding to trainer) ──────────────

def _default_normalise(text: str) -> str:

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────

class TextIterator:

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        min_chars: int = 50,
        max_chars: int = 1_000_000,
        dedup: bool = False,
        normalise_fn: Optional[Callable[[str], str]] = _default_normalise,
        log_every: int = 10_000,
        shuffle_shards: bool = False,
        seed: int = 42,
    ):
        self.dataset_dir  = Path(dataset_dir)
        self.min_chars    = min_chars
        self.max_chars    = max_chars
        self.dedup        = dedup
        self.normalise_fn = normalise_fn
        self.log_every    = log_every
        self.shuffle_shards = shuffle_shards
        self.seed         = seed

        self._shards = self._discover_shards()
        if not self._shards:
            raise FileNotFoundError(
                f"No .parquet files found in '{self.dataset_dir}'. "
                "Run the dataset_worker first."
            )
        log.info("TextIterator: found %d shards in '%s'", len(self._shards), self.dataset_dir)

    # ── Shard discovery ───────────────────────────────────────────────────────

    def _discover_shards(self) -> list[Path]:
        shards = sorted(self.dataset_dir.glob("shard_*.parquet"))
        if self.shuffle_shards:
            import random
            rng = random.Random(self.seed)
            rng.shuffle(shards)
        return shards

    # ── Core iteration ────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[str]:
        """Yield clean text strings."""
        seen: set[str] = set() if self.dedup else set()  # type: ignore
        total = 0
        skipped = 0

        for shard_idx, shard_path in enumerate(self._shards):
            shard_total   = 0
            shard_skipped = 0

            try:
                # Read only the "text" column — avoids loading metadata_json etc.
                table = pq.read_table(shard_path, columns=["text"])
            except Exception as exc:
                log.error("Failed to read shard %s: %s — skipping.", shard_path.name, exc)
                continue

            for text in table.column("text").to_pylist():
                if not text or not isinstance(text, str):
                    shard_skipped += 1
                    skipped += 1
                    continue

                # Apply normalisation
                if self.normalise_fn:
                    text = self.normalise_fn(text)

                # Length filter
                n = len(text)
                if n < self.min_chars or n > self.max_chars:
                    shard_skipped += 1
                    skipped += 1
                    continue

                # Optional exact dedup
                if self.dedup:
                    fp = hashlib.sha256(text.encode()).hexdigest()
                    if fp in seen:
                        shard_skipped += 1
                        skipped += 1
                        continue
                    seen.add(fp)

                total += 1
                shard_total += 1

                if self.log_every and total % self.log_every == 0:
                    log.info(
                        "TextIterator: yielded %d texts  skipped %d  "
                        "(shard %d/%d)",
                        total, skipped, shard_idx + 1, len(self._shards),
                    )

                yield text

            log.debug(
                "Shard %s: yielded %d  skipped %d",
                shard_path.name, shard_total, shard_skipped,
            )

        log.info(
            "TextIterator: finished — total yielded=%d  total skipped=%d  shards=%d",
            total, skipped, len(self._shards),
        )

    # ── Metadata iteration ────────────────────────────────────────────────────

    def with_metadata(self) -> Iterator[tuple[str, dict]]:
        """
        Yield (text, metadata_dict) pairs.
        metadata_dict contains: url, title, language, quality_score, etc.
        """
        import json

        for shard_path in self._shards:
            try:
                table = pq.read_table(
                    shard_path,
                    columns=["text", "source", "language", "quality_score", "metadata_json"],
                )
            except Exception as exc:
                log.error("Failed to read shard %s: %s — skipping.", shard_path.name, exc)
                continue

            texts         = table.column("text").to_pylist()
            sources       = table.column("source").to_pylist()
            languages     = table.column("language").to_pylist()
            quality_scores= table.column("quality_score").to_pylist()
            metadata_jsons= table.column("metadata_json").to_pylist()

            for text, source, lang, qs, meta_raw in zip(
                texts, sources, languages, quality_scores, metadata_jsons
            ):
                if not text:
                    continue
                if self.normalise_fn:
                    text = self.normalise_fn(text)
                if len(text) < self.min_chars:
                    continue

                try:
                    meta = json.loads(meta_raw) if meta_raw else {}
                except json.JSONDecodeError:
                    meta = {}

                meta.update({
                    "source":        source,
                    "language":      lang,
                    "quality_score": qs,
                })
                yield text, meta

    # ── Utility methods ───────────────────────────────────────────────────────

    def count(self) -> int:
        """Count total valid texts without loading text content into RAM."""
        total = 0
        for shard_path in self._shards:
            try:
                # Read only a tiny column to count rows cheaply
                table = pq.read_table(shard_path, columns=["char_count"])
                # Apply the same min/max filter using the stored char_count
                col = table.column("char_count").to_pylist()
                total += sum(
                    1 for c in col
                    if c is not None and self.min_chars <= c <= self.max_chars
                )
            except Exception as exc:
                log.error("count(): failed to read %s: %s", shard_path.name, exc)
        return total

    def shard_stats(self) -> list[dict]:
        """Return per-shard statistics without loading text."""
        stats = []
        for shard_path in self._shards:
            try:
                table = pq.read_table(
                    shard_path,
                    columns=["char_count", "word_count", "token_estimate", "language"],
                )
                n         = len(table)
                chars     = table.column("char_count").to_pylist()
                words     = table.column("word_count").to_pylist()
                tokens    = table.column("token_estimate").to_pylist()
                langs     = table.column("language").to_pylist()

                from collections import Counter
                lang_dist = dict(Counter(l for l in langs if l))

                stats.append({
                    "shard":           shard_path.name,
                    "documents":       n,
                    "total_chars":     sum(c for c in chars if c),
                    "total_words":     sum(w for w in words if w),
                    "total_tokens":    sum(t for t in tokens if t),
                    "avg_chars":       round(sum(c for c in chars if c) / max(n, 1)),
                    "lang_distribution": lang_dist,
                })
            except Exception as exc:
                log.error("shard_stats(): failed on %s: %s", shard_path.name, exc)
        return stats

    def __len__(self) -> int:
        return self.count()
