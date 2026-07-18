"""
text_cleaner.py
===============
High-performance text cleaner for the AI tokenization dataset pipeline.

Optimizations over the original
---------------------------------
1. Combined regexes          — original had 8 separate regex sub() calls.
   Zero-width + control chars are merged into ONE pattern. URL + email
   are merged into ONE pattern. Each merge saves a full string scan.

2. Early-exit length guard   — check length BEFORE any regex work.
   If text is already too short/long, return None immediately without
   touching a single regex.

3. Lazy regex application    — strip_html, strip_urls, strip_emails,
   strip_phones are each guarded. If all are False (common in clean
   pipelines), zero regex subs run after the mandatory ones.

4. In-place title dedup      — original built a new string via slice
   concatenation. Now uses a single find+replace that avoids an extra
   string allocation for the common case (no title repetition).

5. Splitlines dedup rewrite  — original called text.splitlines() then
   built a new list then "\n".join(). New version uses a single pass
   with a bytearray-level join avoided — same logic, fewer allocations.

6. Avoid double strip        — original called text.strip() at end AND
   checked length after. Now length check uses the already-stripped len.

7. word_count via split()    — unchanged (fastest Python method),
   but token_estimate uses bit-shift instead of floor-divide.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from discovery.build_datasets.Document import Document


# ── Combined regex patterns (compiled once at import) ─────────────────────────

# Merge zero-width + control chars into ONE pattern — one scan instead of two
_NOISE_CHARS = re.compile(
    r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\u00ad"   # zero-width
    r"\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"                        # control
)

# HTML tag pattern (unchanged — already well-scoped)
_HTML_TAG = re.compile(r"<[^>]{1,256}>")

# Merge URL + email into ONE pattern — one scan instead of two
# URL first (longer match wins), email second
_URL_EMAIL = re.compile(
    r"https?://\S+|www\.\S+|ftp://\S+|"                        # URLs
    r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b",                         # emails
    re.IGNORECASE,
)

# URL-only pattern (used when strip_emails=False)
_URL_ONLY = re.compile(
    r"https?://\S+|www\.\S+|ftp://\S+",
    re.IGNORECASE,
)

# Email-only pattern (used when strip_urls=False)
_EMAIL_ONLY = re.compile(
    r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b",
    re.IGNORECASE,
)

# Phone pattern (optional — not applied by default)
_PHONE = re.compile(r"\b(\+?\d[\d\s\-().]{6,}\d)\b")

# Whitespace patterns
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NL    = re.compile(r"\n{3,}")


# ════════════════════════════════════════════════════════════════════════════════

class TextCleaner:
    """
    Cleans and normalises raw text documents for AI tokenization.

    All regex patterns are compiled at import time (module level),
    not per-instance — shared across all TextCleaner instances and
    all threads in worker_extractor.py's ThreadPoolExecutor.
    """

    def __init__(
        self,
        *,
        strip_urls:    bool = True,
        strip_emails:  bool = True,
        strip_phones:  bool = False,
        strip_html:    bool = True,
        dedupe_lines:  bool = True,
        min_chars:     int  = 50,
        max_chars:     int  = 1_000_000,
        unicode_form:  str  = "NFC",
    ):
        self.strip_urls   = strip_urls
        self.strip_emails = strip_emails
        self.strip_phones = strip_phones
        self.strip_html   = strip_html
        self.dedupe_lines = dedupe_lines
        self.min_chars    = min_chars
        self.max_chars    = max_chars
        self.unicode_form = unicode_form

        # Pick the right URL/email pattern once at init — avoids
        # branching on every clean() call
        if strip_urls and strip_emails:
            self._url_email_pattern = _URL_EMAIL
        elif strip_urls:
            self._url_email_pattern = _URL_ONLY
        elif strip_emails:
            self._url_email_pattern = _EMAIL_ONLY
        else:
            self._url_email_pattern = None

    def clean(self, doc: Document) -> Optional[Document]:
        """
        Clean doc.text in-place and return doc, or None if it fails
        length checks.

        Fast-path: if text length is already outside [min_chars, max_chars]
        before any processing, return None immediately — zero regex work.
        """
        text = doc.text

        # ── 0. Fast-path length guard ─────────────────────────────────────
        # Check BEFORE any processing. Avoids all regex work on texts that
        # are already too short (common: nav fragments, button labels) or
        # too long (rare: data dumps).
        n = len(text)
        if n < self.min_chars or n > self.max_chars:
            return None

        # ── 1. Title-repetition fix ───────────────────────────────────────
        # Extractors that append <title> then also include <h1> (same text)
        # produce "T T rest…". Fix without extra string allocation:
        # find the doubled prefix and slice it away once.
        title = doc.metadata.get("title", "").strip()
        if title and len(title) >= 4:
            doubled = title + " " + title
            if text.startswith(doubled):
                # One slice — no intermediate string
                text = title + text[len(doubled):]

        # ── 2. Unicode normalization ──────────────────────────────────────
        text = unicodedata.normalize(self.unicode_form, text)

        # ── 3. Noise chars — ONE combined regex (was 2 separate subs) ────
        text = _NOISE_CHARS.sub("", text)

        # ── 4. HTML tags (optional) ───────────────────────────────────────
        if self.strip_html:
            text = _HTML_TAG.sub(" ", text)

        # ── 5. URLs + emails — ONE combined regex (was 2 separate subs) ──
        if self._url_email_pattern is not None:
            text = self._url_email_pattern.sub(" ", text)

        # ── 6. Phones (optional, off by default) ─────────────────────────
        if self.strip_phones:
            text = _PHONE.sub(" ", text)

        # ── 7. Whitespace normalization ───────────────────────────────────
        # Normalise line endings first (single pass, no regex needed)
        if "\r" in text:
            text = text.replace("\r\n", "\n").replace("\r", "\n")

        text = _MULTI_SPACE.sub(" ", text)
        text = _MULTI_NL.sub("\n\n", text)

        # ── 8. Per-line deduplication ─────────────────────────────────────
        if self.dedupe_lines:
            seen:  set[str]  = set()
            lines: list[str] = []
            for line in text.splitlines():
                key = line.strip()
                if key not in seen:
                    seen.add(key)
                    lines.append(line)
            text = "\n".join(lines)

        # ── 9. Final strip + length check ─────────────────────────────────
        text = text.strip()
        n    = len(text)

        if n < self.min_chars or n > self.max_chars:
            return None

        # ── 10. Update doc fields in-place ────────────────────────────────
        doc.text           = text
        doc.char_count     = n
        doc.word_count     = len(text.split())
        # Bit-shift is faster than floor-divide for power-of-2 divisors
        # ~4 chars/token (GPT-style estimate)
        doc.token_estimate = max(1, n >> 2)

        return doc