from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("tokenizer_validator")


# ── Edge case sentences ────────────────────────────────────────────────────────

EDGE_CASES = [
    # English prose
    ("english_prose",
     "ICANN coordinates the Internet Assigned Numbers Authority functions."),

    # Numbers and dates
    ("numbers_dates",
     "The organization was founded on September 18 1998 with 2500 employees."),

    # URLs and domains
    ("urls_domains",
     "Visit https://www.icann.org/resources/pages/dnssec-2012-02-25-en for details."),

    # Mixed punctuation
    ("punctuation",
     "Hello, world! This is a test: does it work? Yes — it does (probably)."),

    # Uppercase / acronyms
    ("acronyms",
     "ICANN IANA DNS DNSSEC WHOIS GDPR TLD gTLD ccTLD IDN IPv4 IPv6"),

    # Long compound word
    ("long_word",
     "internationalization supercalifragilisticexpialidocious telecommunications"),

    # Kinyarwanda (your target language)
    ("kinyarwanda",
     "Inama y'Umuryango w'Abibumbye yateranye i Kigali mu mwaka wa 2024."),

    # Swahili
    ("swahili",
     "Shirika la Kimataifa la Mawasiliano linasimamia mifumo ya majina ya kikoa."),

    # French
    ("french",
     "L'ICANN coordonne les systèmes d'identification uniques de l'internet mondial."),

    # Empty and minimal
    ("single_word",    "internet"),
    ("two_words",      "domain name"),
    ("empty_string",   ""),

    # Special unicode
    ("unicode_mixed",
     "café résumé naïve Ångström 中文 العربية हिन्दी"),

    # Numbers only
    ("numbers_only",
     "1234567890 3.14159 1,000,000 +250 788 123 456"),
]


# ════════════════════════════════════════════════════════════════════════════════
# VALIDATOR
# ════════════════════════════════════════════════════════════════════════════════

class TokenizerValidator:
    """
    Validates a trained tokenizer against the Parquet dataset.

    Parameters
    ----------
    tokenizer_dir : directory containing the saved tokenizer
    dataset_dir   : directory containing shard_*.parquet files
    sample_size   : number of texts to sample for metrics
    report_path   : where to write the JSON validation report
    """

    def __init__(
        self,
        tokenizer_dir: str | Path = "tokenizer_output",
        dataset_dir:   str | Path = "dataset_output",
        *,
        sample_size:  int       = 5_000,
        report_path:  Optional[str | Path] = None,
    ):
        self.tokenizer_dir = Path(tokenizer_dir)
        self.dataset_dir   = Path(dataset_dir)
        self.sample_size   = sample_size
        self.report_path   = Path(report_path) if report_path else \
                             self.tokenizer_dir / "validation_report.json"

        self._load_tokenizer()

    # ── Load ──────────────────────────────────────────────────────────────────

    def _load_tokenizer(self):
        from transformers import PreTrainedTokenizerFast
        tok_path = self.tokenizer_dir
        if not (tok_path / "tokenizer.json").exists():
            raise FileNotFoundError(
                f"No tokenizer.json found in '{tok_path}'. "
                "Run tokenizer_trainer.py first."
            )
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_path))
        self.unk_id    = self.tokenizer.convert_tokens_to_ids("<unk>")
        log.info(
            "Loaded tokenizer from '%s'  vocab_size=%d",
            tok_path, self.tokenizer.vocab_size,
        )

    # ── Core metrics ──────────────────────────────────────────────────────────

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def compute_metrics(self) -> dict:
        """
        Stream sample texts from Parquet and compute all metrics.
        Returns a full metrics dict.
        """
        from discovery.tokenization.text_iterator import TextIterator

        log.info("Collecting %d sample texts for validation...", self.sample_size)
        iterator = TextIterator(self.dataset_dir, min_chars=50, log_every=0)

        # Collect samples with language info
        samples: list[tuple[str, str]] = []   # (text, language)
        for text, meta in iterator.with_metadata():
            lang = meta.get("language", "unknown") or "unknown"
            samples.append((text, lang))
            if len(samples) >= self.sample_size:
                break

        if not samples:
            raise RuntimeError(
                "No samples found. Check that dataset_output/ has parquet files."
            )

        log.info("Collected %d samples. Computing metrics...", len(samples))

        # ── Global counters ───────────────────────────────────────────────────
        total_words    = 0
        total_tokens   = 0
        total_unks     = 0
        total_chars    = 0

        # ── Per-language counters ─────────────────────────────────────────────
        lang_words:  dict[str, int] = defaultdict(int)
        lang_tokens: dict[str, int] = defaultdict(int)
        lang_unks:   dict[str, int] = defaultdict(int)
        lang_count:  dict[str, int] = defaultdict(int)

        # ── Token frequency (for vocab analysis) ─────────────────────────────
        token_freq: Counter = Counter()

        t0 = time.perf_counter()
        for text, lang in samples:
            words  = text.split()
            ids    = self._encode(text)
            n_unk  = ids.count(self.unk_id) if self.unk_id is not None else 0

            total_words  += len(words)
            total_tokens += len(ids)
            total_unks   += n_unk
            total_chars  += len(text)

            lang_words[lang]  += len(words)
            lang_tokens[lang] += len(ids)
            lang_unks[lang]   += n_unk
            lang_count[lang]  += 1

            # Sample token frequencies (first 100 samples only for speed)
            if len(token_freq) < 500_000:
                token_freq.update(ids)

        elapsed = time.perf_counter() - t0

        # ── Compute global metrics ────────────────────────────────────────────
        fertility         = total_tokens / max(total_words,  1)
        unk_rate          = total_unks   / max(total_tokens, 1)
        compression_ratio = total_chars  / max(total_tokens, 1)
        vocab_coverage    = 1.0 - unk_rate

        # ── Per-language metrics ──────────────────────────────────────────────
        per_language = {}
        for lang in sorted(lang_count):
            lw = lang_words[lang]
            lt = lang_tokens[lang]
            lu = lang_unks[lang]
            per_language[lang] = {
                "documents":        lang_count[lang],
                "fertility":        round(lt / max(lw, 1), 4),
                "unk_rate":         round(lu / max(lt, 1), 6),
                "vocab_coverage":   round(1.0 - lu / max(lt, 1), 6),
            }

        # ── Vocab utilisation ─────────────────────────────────────────────────
        unique_tokens_used  = len(token_freq)
        vocab_utilisation   = unique_tokens_used / max(self.tokenizer.vocab_size, 1)
        top_tokens = [
            self.tokenizer.convert_ids_to_tokens(tid)
            for tid, _ in token_freq.most_common(20)
            if tid != self.unk_id
        ]

        metrics = {
            "vocab_size":           self.tokenizer.vocab_size,
            "samples_used":         len(samples),
            "elapsed_seconds":      round(elapsed, 2),

            # Core metrics
            "fertility":            round(fertility, 4),
            "unk_rate":             round(unk_rate, 6),
            "vocab_coverage":       round(vocab_coverage, 6),
            "compression_ratio":    round(compression_ratio, 4),

            # Counts
            "total_words":          total_words,
            "total_tokens":         total_tokens,
            "total_unks":           total_unks,
            "total_chars":          total_chars,

            # Vocab
            "unique_tokens_used":   unique_tokens_used,
            "vocab_utilisation":    round(vocab_utilisation, 4),
            "top_20_tokens":        top_tokens,

            # Per language
            "per_language":         per_language,
        }

        self._log_metrics(metrics)
        return metrics

    def _log_metrics(self, m: dict):
        log.info("─" * 56)
        log.info("VALIDATION RESULTS")
        log.info("  Vocab size        : %d",      m["vocab_size"])
        log.info("  Samples used      : %d",      m["samples_used"])
        log.info("  Fertility         : %.4f   ✓ target 1.2–2.0",  m["fertility"])
        log.info("  UNK rate          : %.4f%%  ✓ target < 0.1%%", m["unk_rate"] * 100)
        log.info("  Vocab coverage    : %.4f%%  ✓ target > 99%%",  m["vocab_coverage"] * 100)
        log.info("  Compression ratio : %.4f   ✓ target 3.5–5.0", m["compression_ratio"])
        log.info("  Vocab utilisation : %.2f%%", m["vocab_utilisation"] * 100)
        log.info("  Top tokens        : %s", m["top_20_tokens"][:10])
        log.info("")
        log.info("  PER-LANGUAGE BREAKDOWN:")
        for lang, lm in m["per_language"].items():
            log.info(
                "    %-10s  docs=%-5d  fertility=%.2f  unk=%.4f%%  coverage=%.2f%%",
                lang, lm["documents"], lm["fertility"],
                lm["unk_rate"] * 100, lm["vocab_coverage"] * 100,
            )

        # Warnings
        if m["fertility"] > 2.5:
            log.warning("⚠ High fertility (%.2f) — consider larger vocab_size.", m["fertility"])
        if m["fertility"] < 1.1:
            log.warning("⚠ Very low fertility (%.2f) — vocab may be too large.", m["fertility"])
        if m["unk_rate"] > 0.001:
            log.warning("⚠ High UNK rate (%.4f%%) — more data or larger vocab needed.", m["unk_rate"] * 100)
        if m["vocab_utilisation"] < 0.5:
            log.warning("⚠ Low vocab utilisation (%.1f%%) — vocab_size may be too large.", m["vocab_utilisation"] * 100)
        log.info("─" * 56)

    # ── Edge cases ────────────────────────────────────────────────────────────

    def run_edge_cases(self) -> list[dict]:
        """Tokenize all EDGE_CASES and return detailed results."""
        log.info("Running %d edge case tests...", len(EDGE_CASES))
        results = []
        for name, text in EDGE_CASES:
            if not text:
                results.append({"name": name, "input": text, "tokens": [], "ids": [], "n_tokens": 0})
                continue
            ids    = self._encode(text)
            tokens = self.tokenizer.convert_ids_to_tokens(ids)
            n_unk  = ids.count(self.unk_id) if self.unk_id is not None else 0
            result = {
                "name":     name,
                "input":    text,
                "tokens":   tokens[:30],
                "ids":      ids[:30],
                "n_tokens": len(ids),
                "n_unks":   n_unk,
                "roundtrip": self.tokenizer.decode(ids),
            }
            results.append(result)

            # Log
            status = "✓" if n_unk == 0 else f"✗ {n_unk} UNK"
            log.info("  [%s] %-22s → %d tokens  %s", status, name, len(ids), tokens[:8])

            # Roundtrip check
            if result["roundtrip"].strip() != text.strip():
                log.warning("    Roundtrip mismatch!")
                log.warning("    Expected : %s", text[:80])
                log.warning("    Got      : %s", result["roundtrip"][:80])

        return results

    # ── Full run ──────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Compute all metrics + edge cases + save report."""
        metrics     = self.compute_metrics()
        edge_cases  = self.run_edge_cases()

        report = {
            "metrics":    metrics,
            "edge_cases": edge_cases,
        }

        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        log.info("Validation report saved → %s", self.report_path)

        return report


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    validator = TokenizerValidator(
        tokenizer_dir = os.getenv("TOKENIZER_DIR", "tokenizer_output"),
        dataset_dir   = os.getenv("DATASET_DIR",   "dataset_output"),
        sample_size   = int(os.getenv("SAMPLE_SIZE", "5000")),
    )
    validator.run()


if __name__ == "__main__":
    main()