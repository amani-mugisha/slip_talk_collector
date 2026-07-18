from __future__ import annotations

import logging
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Sequence

from discovery.build_datasets.Document import Document

logger = logging.getLogger(__name__)

# Compiled once at import time, not on every call to score().
_BULLET_RE = re.compile(r"^\s*[-•*·▪▸►◦‣⁃]\s")


@dataclass(frozen=True)
class QualityWeights:
    """Relative weight of each sub-score in the final average.

    Defaults reproduce the original unweighted (1.0 each) behaviour;
    tune these to emphasize/de-emphasize a signal without touching
    the scoring logic itself.
    """

    symbol: float = 1.0
    digit: float = 1.0
    uppercase: float = 1.0
    avg_word_len: float = 1.0
    unique_ratio: float = 1.0
    bullet: float = 1.0


@dataclass(frozen=True)
class _CharStats:
    """Result of a single pass over a document's characters."""

    n_symbols: int
    n_digits: int
    n_alpha: int
    n_upper: int


class QualityFilter:
    """Heuristic 0-1 text-quality scorer/filter for corpus documents.

    Signals used (each mapped to 0-1, then averaged with ``weights``):
      1. Symbol density        - too many non-alnum chars -> boilerplate/junk
      2. Digit density         - too many digits -> tables, IDs, spam
      3. Uppercase density     - too much upper-case -> shouting/headers
      4. Average word length   - too short/long -> gibberish or code
      5. Unique-word ratio     - low lexical variety -> repeated/templated text
      6. Bullet-line dominance - list-heavy docs -> low prose value

    All character-level counts for (1)-(3) are collected in a single pass
    over the text (see ``_char_stats``) rather than three separate
    comprehensions, and a batch API is provided so a whole corpus can be
    scored/filtered in one call, optionally across multiple processes.
    """

    def __init__(
        self,
        min_score: float = 0.5,
        *,
        min_words: int = 20,
        max_symbol_ratio: float = 0.15,
        max_digit_ratio: float = 0.30,
        max_uppercase_ratio: float = 0.25,
        min_avg_word_len: float = 3.0,
        max_avg_word_len: float = 12.0,
        min_unique_word_ratio: float = 0.20,
        weights: QualityWeights | None = None,
    ):
        if not 0.0 <= min_score <= 1.0:
            raise ValueError(f"min_score must be in [0, 1], got {min_score!r}")
        if min_words < 0:
            raise ValueError(f"min_words must be >= 0, got {min_words!r}")
        for name, value in (
            ("max_symbol_ratio", max_symbol_ratio),
            ("max_digit_ratio", max_digit_ratio),
            ("max_uppercase_ratio", max_uppercase_ratio),
            ("min_unique_word_ratio", min_unique_word_ratio),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value!r}")
        if min_avg_word_len > max_avg_word_len:
            raise ValueError("min_avg_word_len must be <= max_avg_word_len")

        self.min_score = min_score
        self.min_words = min_words
        self.max_symbol_ratio = max_symbol_ratio
        self.max_digit_ratio = max_digit_ratio
        self.max_uppercase_ratio = max_uppercase_ratio
        self.min_avg_word_len = min_avg_word_len
        self.max_avg_word_len = max_avg_word_len
        self.min_unique_word_ratio = min_unique_word_ratio
        self.weights = weights or QualityWeights()

    # ------------------------------------------------------------------
    # Single-document API
    # ------------------------------------------------------------------
    def score(self, doc: Document) -> Document:
        """Compute and attach ``doc.quality_score``. Returns the same doc."""
        doc.quality_score = self._compute_score(doc.text)
        return doc

    def filter(self, doc: Document) -> bool:
        """True iff ``doc`` passes both the length gate and the score gate.

        ``doc`` must already have been scored via :meth:`score` (or as part
        of a batch call) — ``quality_score`` is not recomputed here.
        """
        n_words = len(doc.text.split())
        return n_words >= self.min_words and doc.quality_score >= self.min_score

    # ------------------------------------------------------------------
    # Batch API
    # ------------------------------------------------------------------
    def score_batch(
        self,
        docs: Sequence[Document] | Iterable[Document],
        *,
        n_workers: int | None = None,
    ) -> list[Document]:
        """Score every document in ``docs``.

        For small/medium batches this simply loops (each document is
        already scored in a single pass internally). For large corpora,
        pass ``n_workers > 1`` to fan the batch out across processes —
        useful because the scorer is pure-CPU / GIL-bound text work.
        """
        docs = list(docs)
        if not docs:
            return docs

        if n_workers and n_workers > 1 and len(docs) > 1:
            texts = [d.text for d in docs]
            chunksize = max(1, len(texts) // (n_workers * 4))
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                scores = pool.map(self._compute_score, texts, chunksize=chunksize)
            for d, s in zip(docs, scores):
                d.quality_score = s
        else:
            for d in docs:
                d.quality_score = self._compute_score(d.text)

        return docs

    def filter_batch(
        self,
        docs: Sequence[Document] | Iterable[Document],
        *,
        n_workers: int | None = None,
    ) -> list[Document]:
        """Score then filter a whole corpus in one call."""
        scored = self.score_batch(docs, n_workers=n_workers)
        kept = [d for d in scored if self.filter(d)]
        if scored:
            logger.info(
                "QualityFilter: kept %d/%d documents (%.1f%%)",
                len(kept),
                len(scored),
                100.0 * len(kept) / len(scored),
            )
        return kept

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _char_stats(text: str) -> _CharStats:
        """Classify every character in a single pass.

        Replaces three independent ``sum(1 for c in text if ...)``
        comprehensions (one each for symbols, digits, uppercase) with one
        O(n) loop, halving/thirding the character-level work per document.
        """
        n_symbols = n_digits = n_alpha = n_upper = 0
        for c in text:
            if c.isalpha():
                n_alpha += 1
                if c.isupper():
                    n_upper += 1
            elif c.isdigit():
                n_digits += 1
            elif not c.isspace():
                n_symbols += 1
        return _CharStats(n_symbols, n_digits, n_alpha, n_upper)

    def _compute_score(self, text: str) -> float:
        """Pure function: text -> quality score in [0, 1].

        Kept free of ``Document`` / ``self`` mutation so it can be shipped
        to worker processes via ``ProcessPoolExecutor.map`` unchanged.
        """
        n_chars = len(text)
        if n_chars == 0:
            return 0.0

        words = text.split()
        n_words = len(words)
        if n_words == 0:
            # Whitespace-only text: no words to compute avg length /
            # uniqueness from, so treat as lowest quality rather than
            # dividing by zero.
            return 0.0

        stats = self._char_stats(text)
        w = self.weights

        scored: list[tuple[float, float]] = []  # (weight, sub_score)

        # 1. Symbol ratio
        sym_ratio = stats.n_symbols / n_chars
        scored.append(
            (w.symbol, 1.0 - min(sym_ratio / self.max_symbol_ratio, 1.0) * 0.5)
        )

        # 2. Digit ratio
        dig_ratio = stats.n_digits / n_chars
        scored.append(
            (w.digit, 1.0 - min(dig_ratio / self.max_digit_ratio, 1.0) * 0.5)
        )

        # 3. Uppercase ratio (over alpha chars only)
        if stats.n_alpha:
            up_ratio = stats.n_upper / stats.n_alpha
            scored.append(
                (
                    w.uppercase,
                    1.0 - min(up_ratio / self.max_uppercase_ratio, 1.0) * 0.5,
                )
            )

        # 4. Average word length
        avg_wl = sum(len(word) for word in words) / n_words
        wl_ok = self.min_avg_word_len <= avg_wl <= self.max_avg_word_len
        scored.append((w.avg_word_len, 1.0 if wl_ok else 0.5))

        # 5. Unique word ratio (vocabulary richness)
        unique_ratio = len({word.lower() for word in words}) / n_words
        scored.append(
            (w.unique_ratio, min(unique_ratio / self.min_unique_word_ratio, 1.0))
        )

        # 6. Bullet / list line dominance
        lines = [line for line in text.splitlines() if line.strip()]
        if lines:
            bullet_lines = sum(1 for line in lines if _BULLET_RE.match(line))
            bullet_ratio = bullet_lines / len(lines)
            scored.append((w.bullet, 1.0 - min(bullet_ratio * 2, 1.0) * 0.4))

        weight_sum = sum(weight for weight, _ in scored)
        if weight_sum <= 0:
            return 0.0

        raw = sum(weight * value for weight, value in scored) / weight_sum
        return round(min(max(raw, 0.0), 1.0), 4)