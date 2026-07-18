from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("tokenizer_trainer")


# ── Special tokens used across all algorithms ─────────────────────────────────

SPECIAL_TOKENS = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]


# ════════════════════════════════════════════════════════════════════════════════
# TOKENIZER TRAINER
# ════════════════════════════════════════════════════════════════════════════════

class TokenizerTrainer:
    """
    Trains a subword tokenizer from a TextIterator and saves it.

    Parameters
    ----------
    dataset_dir     : directory with shard_*.parquet files
    tokenizer_dir   : where to save the trained tokenizer
    vocab_size      : target vocabulary size (default 32 000)
    min_frequency   : minimum token frequency to include (default 2)
    algorithm       : "bpe" | "wordpiece" | "unigram"
    show_progress   : show training progress bar (default True)
    """

    def __init__(
        self,
        dataset_dir:   str | Path = "dataset_output",
        tokenizer_dir: str | Path = "tokenizer_output",
        *,
        vocab_size:     int  = 32_000,
        min_frequency:  int  = 2,
        algorithm:      str  = "bpe",
        show_progress:  bool = True,
    ):
        self.dataset_dir   = Path(dataset_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.vocab_size    = vocab_size
        self.min_frequency = min_frequency
        self.algorithm     = algorithm.lower()
        self.show_progress = show_progress

        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)

        if self.algorithm not in ("bpe", "wordpiece", "unigram"):
            raise ValueError(f"Unknown algorithm '{algorithm}'. Choose: bpe, wordpiece, unigram")

    # ── Build tokenizer ───────────────────────────────────────────────────────

    def _build_tokenizer(self):
        """Construct the tokenizer + trainer objects for the chosen algorithm."""
        from tokenizers import Tokenizer, pre_tokenizers, decoders, processors
        from tokenizers import models, trainers, normalizers

        if self.algorithm == "bpe":
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import ByteLevel
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder
            from tokenizers.processors import ByteLevel as ByteLevelProcessor

            tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
            tokenizer.decoder = ByteLevelDecoder()
            tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)

            trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                min_frequency=self.min_frequency,
                special_tokens=SPECIAL_TOKENS,
                show_progress=self.show_progress,
            )

        elif self.algorithm == "wordpiece":
            from tokenizers.models import WordPiece
            from tokenizers.trainers import WordPieceTrainer
            from tokenizers.pre_tokenizers import Whitespace
            from tokenizers.normalizers import BertNormalizer

            tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
            tokenizer.normalizer = BertNormalizer(lowercase=False)
            tokenizer.pre_tokenizer = Whitespace()

            trainer = WordPieceTrainer(
                vocab_size=self.vocab_size,
                min_frequency=self.min_frequency,
                special_tokens=SPECIAL_TOKENS,
                show_progress=self.show_progress,
            )

        else:  # unigram
            from tokenizers.models import Unigram
            from tokenizers.trainers import UnigramTrainer
            from tokenizers.pre_tokenizers import Metaspace

            tokenizer = Tokenizer(Unigram())
            tokenizer.pre_tokenizer = Metaspace()

            trainer = UnigramTrainer(
                vocab_size=self.vocab_size,
                special_tokens=SPECIAL_TOKENS,
                unk_token="<unk>",
                show_progress=self.show_progress,
            )

        return tokenizer, trainer

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self) -> "Tokenizer":  # type: ignore[name-defined]
        """
        Stream all texts through the trainer and return the trained tokenizer.
        """
        # Import here so the module can be imported without tokenizers installed
        from discovery.tokenization.text_iterator import TextIterator

        log.info("=" * 60)
        log.info("Tokenizer Trainer")
        log.info("  Algorithm   : %s", self.algorithm.upper())
        log.info("  Vocab size  : %d", self.vocab_size)
        log.info("  Min freq    : %d", self.min_frequency)
        log.info("  Dataset     : %s", self.dataset_dir)
        log.info("  Output      : %s", self.tokenizer_dir)
        log.info("=" * 60)

        # ── Shard stats before training ──────────────────────────────────────
        iterator = TextIterator(
            self.dataset_dir,
            min_chars=50,
            dedup=False,         # builder already deduped; skip for speed
            log_every=50_000,
        )
        stats = iterator.shard_stats()
        total_docs   = sum(s["documents"]    for s in stats)
        total_tokens = sum(s["total_tokens"] for s in stats)
        log.info(
            "Dataset: %d shards  %d documents  ~%d estimated tokens",
            len(stats), total_docs, total_tokens,
        )
        for s in stats:
            log.info(
                "  %s  docs=%d  avg_chars=%d  langs=%s",
                s["shard"], s["documents"], s["avg_chars"],
                s["lang_distribution"],
            )

        # ── Build model + trainer ─────────────────────────────────────────────
        tokenizer, trainer = self._build_tokenizer()

        # ── Train ─────────────────────────────────────────────────────────────
        log.info("Training %s tokenizer...", self.algorithm.upper())
        t0 = time.perf_counter()

        # Re-create iterator for actual training pass
        train_iterator = TextIterator(
            self.dataset_dir,
            min_chars=50,
            dedup=False,
            log_every=50_000,
        )
        tokenizer.train_from_iterator(
            iter(train_iterator),
            trainer=trainer,
            length=total_docs,   # enables progress bar ETA
        )

        elapsed = time.perf_counter() - t0
        log.info("Training complete in %.1f seconds.", elapsed)
        log.info("Actual vocab size: %d", tokenizer.get_vocab_size())

        self._tokenizer = tokenizer
        return tokenizer

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self, n_samples: int = 2_000) -> dict:
        """
        Run validation metrics on the trained tokenizer.

        Metrics
        -------
        fertility       : avg tokens per word (good: 1.2 – 2.0)
        unk_rate        : fraction of tokens that are <unk> (good: < 0.1%)
        vocab_coverage  : fraction of words fully represented (good: > 99%)
        """
        from discovery.tokenization.text_iterator import TextIterator

        if not hasattr(self, "_tokenizer"):
            raise RuntimeError("Call train() before validate().")

        tokenizer = self._tokenizer
        unk_id    = tokenizer.token_to_id("<unk>")

        log.info("Validating tokenizer on %d sample texts...", n_samples)

        iterator = TextIterator(self.dataset_dir, min_chars=50, log_every=0)
        samples  = []
        for text in iterator:
            samples.append(text)
            if len(samples) >= n_samples:
                break

        total_words  = 0
        total_tokens = 0
        total_unks   = 0

        for text in samples:
            words  = text.split()
            enc    = tokenizer.encode(text)
            ids    = enc.ids
            total_words  += len(words)
            total_tokens += len(ids)
            total_unks   += ids.count(unk_id) if unk_id is not None else 0

        fertility    = total_tokens / max(total_words, 1)
        unk_rate     = total_unks   / max(total_tokens, 1)
        vocab_coverage = 1.0 - unk_rate

        metrics = {
            "vocab_size":     tokenizer.get_vocab_size(),
            "fertility":      round(fertility, 4),
            "unk_rate":       round(unk_rate, 6),
            "vocab_coverage": round(vocab_coverage, 6),
            "total_words":    total_words,
            "total_tokens":   total_tokens,
            "total_unks":     total_unks,
            "samples_used":   len(samples),
        }

        log.info("─" * 50)
        log.info("VALIDATION RESULTS")
        log.info("  Vocab size     : %d",     metrics["vocab_size"])
        log.info("  Fertility      : %.4f  (target: 1.2 – 2.0)", metrics["fertility"])
        log.info("  UNK rate       : %.4%%  (target: < 0.1%%)", metrics["unk_rate"] * 100)
        log.info("  Vocab coverage : %.4%%  (target: > 99%%)",  metrics["vocab_coverage"] * 100)
        log.info("─" * 50)

        # Fertility warning
        if fertility > 2.5:
            log.warning(
                "High fertility (%.2f) — vocabulary may be too small or "
                "training data too small. Consider increasing vocab_size.", fertility
            )
        elif fertility < 1.1:
            log.warning(
                "Very low fertility (%.2f) — vocabulary may be too large "
                "or training data too homogeneous.", fertility
            )

        if unk_rate > 0.001:
            log.warning(
                "High UNK rate (%.4f%%) — consider more training data "
                "or larger vocab_size.", unk_rate * 100
            )

        # Spot-check examples
        spot_checks = [
            "ICANN coordinates the internet naming system globally.",
            "The Domain Name System resolves human-readable addresses.",
            "Tokenization splits text into subword units for neural networks.",
            "Inama y'Umuryango w'Abibumbye yateranye i Kigali.",  # Kinyarwanda
        ]
        log.info("SPOT CHECKS")
        for text in spot_checks:
            enc = tokenizer.encode(text)
            log.info("  Input  : %s", text)
            log.info("  Tokens : %s", enc.tokens[:20])
            log.info("  IDs    : %s", enc.ids[:20])
            log.info("")

        return metrics

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(self) -> Path:
        """
        Save the trained tokenizer in two formats:
          1. Native tokenizers.json  (fast, used by HuggingFace)
          2. HuggingFace PreTrainedTokenizerFast  (use with transformers)
        """
        from transformers import PreTrainedTokenizerFast

        if not hasattr(self, "_tokenizer"):
            raise RuntimeError("Call train() before save().")

        tokenizer = self._tokenizer

        # ── 1. Save native format ────────────────────────────────────────────
        native_path = self.tokenizer_dir / "tokenizer.json"
        tokenizer.save(str(native_path))
        log.info("Saved native tokenizer → %s", native_path)

        # ── 2. Save as HuggingFace PreTrainedTokenizerFast ───────────────────
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(native_path),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            padding_side="right",
            truncation_side="right",
            model_max_length=2048,
        )
        hf_tokenizer.save_pretrained(str(self.tokenizer_dir))
        log.info("Saved HuggingFace tokenizer → %s/", self.tokenizer_dir)

        # ── 3. Save training config for reproducibility ──────────────────────
        config = {
            "algorithm":      self.algorithm,
            "vocab_size":     self.vocab_size,
            "min_frequency":  self.min_frequency,
            "special_tokens": SPECIAL_TOKENS,
            "dataset_dir":    str(self.dataset_dir),
        }
        config_path = self.tokenizer_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        log.info("Saved training config → %s", config_path)

        # ── Summary ──────────────────────────────────────────────────────────
        log.info("=" * 60)
        log.info("Tokenizer saved to: %s", self.tokenizer_dir)
        log.info("Files:")
        for p in sorted(self.tokenizer_dir.iterdir()):
            log.info("  %s  (%.1f KB)", p.name, p.stat().st_size / 1024)
        log.info("=" * 60)

        return self.tokenizer_dir

    # ── One-shot convenience ──────────────────────────────────────────────────

    def run(self) -> dict:
        """Train → Validate → Save in one call. Returns validation metrics."""
        self.train()
        metrics = self.validate()
        self.save()
        return metrics


# ════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

def main():
    dataset_dir   = os.getenv("DATASET_DIR",   "dataset_output")
    tokenizer_dir = os.getenv("TOKENIZER_DIR", "tokenizer_output")
    vocab_size    = int(os.getenv("VOCAB_SIZE",    "32000"))
    min_frequency = int(os.getenv("MIN_FREQUENCY", "2"))
    algorithm     = os.getenv("ALGORITHM",         "bpe")

    trainer = TokenizerTrainer(
        dataset_dir=dataset_dir,
        tokenizer_dir=tokenizer_dir,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        algorithm=algorithm,
    )

    metrics = trainer.run()
    log.info("Done. Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
