"""
seeker.py
=========
High-performance URL processing pipeline: parse → validate → normalize.

Fix applied
-----------
The original seeker.py treated parse() as a parser object and called
methods like parser.find_scheme(), parser.host() etc. — but parse()
already returns a fully populated url_model directly.

  WRONG (original):
      parser = parse(raw_url)
      url = url_model(scheme=parser.find_scheme(), host=parser.host(), ...)

  CORRECT:
      url = parse(raw_url)   # url_model already has .scheme, .host, etc.

All other optimizations from before are kept:
  - Fast-path rejection (scheme, extension, length checks)
  - LRU cache (duplicate URLs skipped instantly)
  - Exception safety (no crashes on malformed input)
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

from discovery.sitemap_fetcher.seeker.url_parser import parse
from discovery.sitemap_fetcher.seeker.validator import Validate
from discovery.sitemap_fetcher.seeker.normalizer import URLNormazer


# ── Fast-path constants ───────────────────────────────────────────────────────

_BINARY_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv",
    ".woff", ".woff2", ".ttf", ".eot",
    ".css", ".js", ".json", ".xml",
    ".exe", ".dmg", ".apk",
})

_MAX_URL_LENGTH = 2048
_SCHEME_RE      = re.compile(r"^https?://", re.IGNORECASE)
_EXTENSION_RE   = re.compile(r"\.([a-z0-9]{1,5})(?:\?|#|$)", re.IGNORECASE)


def _fast_reject(raw_url: str) -> bool:
    """Return True if URL should be rejected before any object creation."""
    if len(raw_url) > _MAX_URL_LENGTH:
        return True
    if not _SCHEME_RE.match(raw_url):
        return True
    path_end = raw_url.find("?")
    if path_end == -1:
        path_end = raw_url.find("#")
    path = raw_url[:path_end] if path_end != -1 else raw_url
    m = _EXTENSION_RE.search(path)
    if m and ("." + m.group(1).lower()) in _BINARY_EXTENSIONS:
        return True
    return False


# ════════════════════════════════════════════════════════════════════════════════

class seekingProcessing:
    """
    URL processing pipeline: parse → validate → normalize.

    parse() returns a url_model directly with fields:
        .scheme, .host, .path, .port, .query, .fragment,
        .user, .password, .raw, .normalized
    """

    @staticmethod
    @lru_cache(maxsize=32_768)
    def processes(raw_url: str) -> Optional[object]:
        """
        Process a raw URL. Returns normalized url_model or None if invalid.
        LRU cached — duplicate URLs return instantly after the first call.
        """
        if not raw_url or not isinstance(raw_url, str):
            return None

        raw_url = raw_url.strip()
        if not raw_url:
            return None

        # Fast-path: reject without creating any objects
        if _fast_reject(raw_url):
            return None

        # parse() returns a url_model directly — no need to call methods on it
        try:
            url = parse(raw_url)
        except Exception:
            return None

        # Validate the url_model
        try:
            validation = Validate.validate(url)
            if not validation.is_valid:
                return None
        except Exception:
            return None

        # Normalize
        try:
            return URLNormazer.normalize(url)
        except Exception:
            return None

    @staticmethod
    def clear_cache() -> None:
        seekingProcessing.processes.cache_clear()

    @staticmethod
    def cache_info() -> str:
        info     = seekingProcessing.processes.cache_info()
        hit_rate = info.hits / max(info.hits + info.misses, 1) * 100
        return (
            f"hits={info.hits}  misses={info.misses}  "
            f"hit_rate={hit_rate:.1f}%  "
            f"size={info.currsize}/{info.maxsize}"
        )