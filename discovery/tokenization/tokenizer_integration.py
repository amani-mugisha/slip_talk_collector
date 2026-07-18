from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Generator, Iterator, Optional

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("tokenizer_integration")


class TokenizerIntegration:
    """
    Clean wrapper around the trained HuggingFace tokenizer.

    Handles: encoding, decoding, batching, padding, truncation,
    special token management, and sequence length statistics.

    Parameters
    ----------
    tokenizer_dir : path to the saved tokenizer
    max_length    : maximum sequence length (default 512)
    """

    def __init__(
        self,
        tokenizer_dir: str | Path = "tokenizer_output",
        *,
        max_length: int = 512,
    ):
        self.tokenizer_dir = Path(tokenizer_dir)
        self.max_length    = max_length
        self._load()

    # ── Load ──────────────────────────────────────────────────────────────────

    def _load(self):
        from transformers import PreTrainedTokenizerFast
        tok_path = self.tokenizer_dir

        if not (tok_path / "tokenizer.json").exists():
            raise FileNotFoundError(
                f"No tokenizer.json in '{tok_path}'. "
                "Run tokenizer_trainer.py first."
            )

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_path))

        # Ensure pad_token is set — required for batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            log.warning("pad_token was None — set to eos_token (%s)", self.tokenizer.eos_token)

        log.info(
            "Tokenizer loaded from '%s'  "
            "vocab=%d  max_len=%d  pad='%s'  bos='%s'  eos='%s'",
            tok_path,
            self.tokenizer.vocab_size,
            self.max_length,
            self.tokenizer.pad_token,
            self.tokenizer.bos_token,
            self.tokenizer.eos_token,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def bos_id(self) -> int:
        return self.tokenizer.bos_token_id

    @property
    def eos_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def unk_id(self) -> int:
        return self.tokenizer.unk_token_id

    # ── Single text encoding ──────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncate: bool = True,
    ) -> list[int]:
        """Encode a single text → list of token IDs."""
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            truncation=truncate,
            max_length=self.max_length if truncate else None,
        )

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token IDs → text string."""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def tokenize(self, text: str) -> list[str]:
        """Return token strings (not IDs) for a text."""
        return self.tokenizer.tokenize(text)

    # ── Batch encoding ────────────────────────────────────────────────────────

    def encode_batch(
        self,
        texts: list[str],
        *,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
    ) -> dict:
        """
        Encode a batch of texts with padding and truncation.

        Parameters
        ----------
        texts          : list of text strings
        padding        : pad all sequences to same length
        truncation     : truncate sequences over max_length
        return_tensors : "pt" for PyTorch, "tf" for TensorFlow,
                         "np" for NumPy, None for plain lists

        Returns
        -------
        dict with keys: input_ids, attention_mask
        """
        return self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )

    # ── Streaming batch generator (for training loops) ────────────────────────

    def batch_generator(
        self,
        texts: Iterator[str],
        batch_size: int = 32,
        *,
        return_tensors: str = "pt",
        drop_last: bool = False,
    ) -> Generator[dict, None, None]:
        """
        Yield tokenized batches from a text iterator.
        Designed to plug directly into a training loop.

        Usage
        -----
            from discovery.tokenization.text_iterator import TextIterator
            texts = TextIterator("dataset_output")
            for batch in integration.batch_generator(texts, batch_size=32):
                input_ids      = batch["input_ids"]       # (B, T)
                attention_mask = batch["attention_mask"]  # (B, T)
                ...
        """
        buffer: list[str] = []
        for text in texts:
            buffer.append(text)
            if len(buffer) >= batch_size:
                yield self.encode_batch(
                    buffer,
                    return_tensors=return_tensors,
                )
                buffer = []

        # Last partial batch
        if buffer and not drop_last:
            yield self.encode_batch(buffer, return_tensors=return_tensors)

    # ── Language model labels ─────────────────────────────────────────────────

    def make_lm_batch(
        self,
        texts: list[str],
        *,
        return_tensors: str = "pt",
    ) -> dict:
        """
        Encode texts and create causal LM labels.
        Labels are the same as input_ids but with padding positions set to -100
        (the standard ignore index in PyTorch CrossEntropyLoss).

        Returns
        -------
        dict with keys: input_ids, attention_mask, labels
        """
        batch = self.encode_batch(texts, return_tensors=return_tensors)

        # Build labels: copy input_ids, mask padding
        try:
            import torch
            import copy; labels = copy.deepcopy(batch["input_ids"])
            labels[labels == self.pad_id] = -100
            batch["labels"] = labels
        except ImportError:
            # Fallback: plain Python lists
            input_ids = batch["input_ids"]
            batch["labels"] = [
                [tid if tid != self.pad_id else -100 for tid in row]
                for row in input_ids
            ]

        return batch

    # ── Sequence length analysis ──────────────────────────────────────────────

    def analyze_lengths(
        self,
        texts: list[str],
        *,
        percentiles: list[int] = [50, 75, 90, 95, 99],
    ) -> dict:
        """
        Analyze token sequence lengths across a list of texts.
        Useful for choosing max_length for your model.
        """
        lengths = [len(self.encode(t, truncate=False)) for t in texts]
        lengths.sort()
        n = len(lengths)

        result = {
            "count":   n,
            "min":     lengths[0]  if lengths else 0,
            "max":     lengths[-1] if lengths else 0,
            "mean":    round(sum(lengths) / max(n, 1), 1),
            "percentiles": {},
        }
        for p in percentiles:
            idx = min(int(n * p / 100), n - 1)
            result["percentiles"][f"p{p}"] = lengths[idx]

        log.info("Sequence length analysis (%d texts):", n)
        log.info("  min=%d  max=%d  mean=%.1f", result["min"], result["max"], result["mean"])
        for k, v in result["percentiles"].items():
            log.info("  %s = %d tokens", k, v)

        # Recommendation
        p95 = result["percentiles"].get("p95", 512)
        recommended = min(2 ** (p95.bit_length()), 2048)  # next power of 2
        log.info("  → Recommended max_length: %d (covers p95)", recommended)
        result["recommended_max_length"] = recommended

        return result


# ════════════════════════════════════════════════════════════════════════════════
# DATASET TOKENIZER  (batch-tokenize entire Parquet dataset)
# ════════════════════════════════════════════════════════════════════════════════

class DatasetTokenizer:
    """
    Tokenize an entire Parquet dataset and save as tokenized Parquet shards.

    The output shards contain: input_ids, attention_mask, length columns —
    ready to be consumed directly by a PyTorch DataLoader or HuggingFace Trainer.

    Parameters
    ----------
    integration   : a loaded TokenizerIntegration instance
    dataset_dir   : source Parquet directory (from dataset_builder)
    output_dir    : where to write tokenized Parquet shards
    batch_size    : texts per batch during tokenization
    """

    def __init__(
        self,
        integration:  TokenizerIntegration,
        dataset_dir:  str | Path = "dataset_output",
        output_dir:   str | Path = "tokenized_output",
        *,
        batch_size:   int = 256,
    ):
        self.integration = integration
        self.dataset_dir = Path(dataset_dir)
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size  = batch_size

    def run(self) -> dict:
        """
        Tokenize all texts and write tokenized_shard_*.parquet files.
        Returns summary statistics.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        from discovery.tokenization.text_iterator import TextIterator

        iterator  = TextIterator(self.dataset_dir, min_chars=50, log_every=50_000)
        gen       = self.integration.batch_generator(
            iter(iterator),
            batch_size=self.batch_size,
            return_tensors=None,   # plain lists — no PyTorch dependency here
            drop_last=False,
        )

        schema = pa.schema([
            ("input_ids",       pa.list_(pa.int32())),
            ("attention_mask",  pa.list_(pa.int8())),
            ("length",          pa.int32()),
        ])

        shard_idx    = 0
        total_seqs   = 0
        total_tokens = 0
        t0           = time.perf_counter()

        for batch in gen:
            ids   = batch["input_ids"]
            masks = batch["attention_mask"]
            lens  = [len(row) for row in ids]

            table = pa.table({
                "input_ids":      ids,
                "attention_mask": masks,
                "length":         lens,
            }, schema=schema)

            out = self.output_dir / f"tokenized_shard_{shard_idx:05d}.parquet"
            pq.write_table(table, out, compression="zstd")

            total_seqs   += len(ids)
            total_tokens += sum(lens)
            shard_idx    += 1

        elapsed = time.perf_counter() - t0
        stats = {
            "shards":        shard_idx,
            "total_seqs":    total_seqs,
            "total_tokens":  total_tokens,
            "elapsed_s":     round(elapsed, 2),
            "tokens_per_sec": round(total_tokens / max(elapsed, 0.001)),
        }
        log.info(
            "DatasetTokenizer done: %d shards  %d sequences  %d tokens  %.1fs",
            stats["shards"], stats["total_seqs"], stats["total_tokens"], elapsed,
        )
        return stats


# ════════════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST
# ════════════════════════════════════════════════════════════════════════════════

def quick_test(tokenizer_dir: str = "tokenizer_output"):
    """
    Quick sanity check — run before starting model training.
    Verifies: load, encode, decode, batch, roundtrip, special tokens.
    """
    log.info("Running quick integration test...")
    integration = TokenizerIntegration(tokenizer_dir)

    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            log.info("  ✓ %s", name)
            passed += 1
        else:
            log.error("  ✗ %s  %s", name, detail)
            failed += 1

    # 1. Vocab size
    check("vocab_size > 0", integration.vocab_size > 0, str(integration.vocab_size))

    # 2. Special tokens
    check("bos_token exists", integration.tokenizer.bos_token is not None)
    check("eos_token exists", integration.tokenizer.eos_token is not None)
    check("pad_token exists", integration.tokenizer.pad_token is not None)
    check("unk_token exists", integration.tokenizer.unk_token is not None)

    # 3. Encode / decode roundtrip
    text = "ICANN coordinates the internet naming system."
    ids  = integration.encode(text)
    check("encode returns list",    isinstance(ids, list))
    check("encode non-empty",       len(ids) > 0)
    decoded = integration.decode(ids)
    check("decode roundtrip",       text.lower() in decoded.lower(), repr(decoded))

    # 4. Batch encode
    texts = [
        "Domain Name System manages addressing.",
        "WHOIS provides registration information.",
        "Short.",
    ]
    batch = integration.encode_batch(texts, padding=True, return_tensors=None)
    check("batch has input_ids",        "input_ids" in batch)
    check("batch has attention_mask",   "attention_mask" in batch)
    check("batch length matches",       len(batch["input_ids"]) == len(texts))

    # Check padding — all rows same length
    row_lengths = [len(row) for row in batch["input_ids"]]
    check("batch rows same length", len(set(row_lengths)) == 1, str(row_lengths))

    # 5. Truncation
    long_text = "internet " * 1000
    ids_long  = integration.encode(long_text, truncate=True)
    check("truncation works", len(ids_long) <= integration.max_length, str(len(ids_long)))

    # 6. LM batch labels
    lm = integration.make_lm_batch(texts[:2], return_tensors=None)
    check("lm batch has labels", "labels" in lm)

    # 7. Length analysis
    sample_texts = [
        "Short sentence.",
        "Medium length sentence with a few more words in it.",
        "A longer sentence that has quite a few words " * 5,
    ]
    stats = integration.analyze_lengths(sample_texts)
    check("length analysis runs", "mean" in stats and stats["mean"] > 0)

    log.info("─" * 40)
    log.info("Quick test: %d passed  %d failed", passed, failed)
    if failed > 0:
        log.error("INTEGRATION TEST FAILED — fix errors before training.")
    else:
        log.info("All checks passed — tokenizer is ready for training.")
    return failed == 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    tokenizer_dir = os.getenv("TOKENIZER_DIR", "tokenizer_output")
    dataset_dir   = os.getenv("DATASET_DIR",   "dataset_output")

    # Step 1: quick test
    ok = quick_test(tokenizer_dir)
    if not ok:
        return

    # Step 2: show how to use in a training loop
    log.info("")
    log.info("=" * 56)
    log.info("Example: streaming batches for a training loop")
    log.info("=" * 56)

    integration = TokenizerIntegration(tokenizer_dir)

    from discovery.tokenization.text_iterator import TextIterator
    texts = TextIterator(dataset_dir, min_chars=50, log_every=0)

    batch_count = 0
    total_tokens = 0
    for batch in integration.batch_generator(iter(texts), batch_size=32, return_tensors=None):
        ids = batch["input_ids"]
        total_tokens += sum(len(row) for row in ids)
        batch_count  += 1
        if batch_count >= 3:
            log.info(
                "  batch %d: shape=(%d, %d)  total_tokens_so_far=%d",
                batch_count, len(ids), len(ids[0]) if ids else 0, total_tokens,
            )
            break

    log.info("")
    log.info("Tokenizer is ready. Use TokenizerIntegration in your training script:")
    log.info("")
    log.info("  from discovery.build_datasets.tokenization.tokenizer_integration import TokenizerIntegration")
    log.info("  integration = TokenizerIntegration('tokenizer_output')")
    log.info("  for batch in integration.batch_generator(texts, batch_size=32, return_tensors='pt'):")
    log.info("      loss = model(**batch).loss")
    log.info("      loss.backward()")


if __name__ == "__main__":
    main()
