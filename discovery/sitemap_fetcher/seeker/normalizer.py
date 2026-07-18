from __future__ import annotations

import logging
import posixpath
from typing import Optional
from urllib.parse import parse_qsl, quote, urlencode

from discovery.sitemap_fetcher.models.urls_model import url_model

logger = logging.getLogger(__name__)

DEFAULT_SCHEME = "https"

DEFAULT_PORTS = {
    "http": 80,
    "https": 443,
}

MIN_PORT = 1
MAX_PORT = 65535

# Characters that are safe to leave unescaped in a path segment.
_PATH_SAFE_CHARS = "/-._~!$&'()*+,;=:@"

class URLNormazer:
    @staticmethod
    def _normalize_scheme(url: url_model) -> str:
        scheme = (url.scheme or DEFAULT_SCHEME).strip().lower()
        return scheme
    
    @staticmethod
    def _normalize_host(url: url_model) -> str:
        host = (url.host or "").strip().lower()
        if host.endswith(".") and host !=".":
            host = host.rstrip(".")

        return host
    
    @staticmethod
    def _normalize_port(url: url_model, scheme: str) -> Optional[int]:
        if url.port in (None, ""):
            return None
        
        try:
            port = int(url.port)
        except (TypeError, ValueError):
            logger.warning(
                "Non-numeric port %r encountered during normalization; "
                "dropping port from normalized URL.",
                url.port,
            )
            return None
        
        if not (MIN_PORT <= port <= MAX_PORT):
            logger.warning(
                "Out-of-range port %s encountered during normalization; "
                "dropping port from normalized URL.",
                port,
            )
            return None

        if DEFAULT_PORTS.get(scheme) == port:
            return None

        return port
    
    @staticmethod
    def _normalize_path(url: url_model) -> str:
        raw_path = url.path or "/"

        if not raw_path.startswith("/"):
            raw_path = f"/{raw_path}"

        collapsed = posixpath.normpath(raw_path)

        if not collapsed.startswith("/"):
            collapsed = f"/{collapsed}"

        collapsed = "/" + collapsed.lstrip("/")

        safe_path = quote(collapsed, safe=_PATH_SAFE_CHARS + "%")

        # Remove trailing slash except for the root path itself.
        if len(safe_path) > 1 and safe_path.endswith("/"):
            safe_path = safe_path[:-1]

        return safe_path

    @staticmethod
    def _normalize_query(raw_query: str) -> str:
        if not raw_query:
            return ""

        params = parse_qsl(raw_query, keep_blank_values=False)

        # Sort by (key, value) for deterministic, order-independent
        # output so that "?b=2&a=1" and "?a=1&b=2" normalize identically.
        params.sort()

        return urlencode(params)

    @classmethod
    def normalize(cls, url: url_model) -> url_model:
        """Normalize `url` in place and return it.

        Canonicalizes scheme/host casing, drops default ports, resolves
        `.`/`..` path segments, collapses duplicate slashes, strips
        trailing slashes (except root), sorts and re-encodes query
        parameters, and discards the fragment (irrelevant for sitemap
        crawling/deduping).

        Mutates and returns the same `url_model` instance; the caller's
        reference is updated, nothing is copied.
        """
        scheme = cls._normalize_scheme(url)
        host = cls._normalize_host(url)
        port = cls._normalize_port(url, scheme)
        path = cls._normalize_path(url)
        query = cls._normalize_query(url.query or "")

        normalized = f"{scheme}://"

        if url.user:
            normalized += quote(url.user, safe="")
            if url.password:
                normalized += f":{quote(url.password, safe='')}"
            normalized += "@"

        normalized += host

        if port is not None:
            normalized += f":{port}"

        normalized += path

        if query:
            normalized += f"?{query}"

        url.scheme = scheme
        url.host = host
        url.port = port
        url.path = path
        url.query = query
        url.fragment = ""
        url.normalized = normalized

        return url