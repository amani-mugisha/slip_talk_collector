"""
diagnose_seeker.py
==================
Run this to find exactly why URLs are being skipped.

    python3 diagnose_seeker.py
"""

import sys
sys.path.insert(0, '.')

test_urls = [
    "https://stackoverflow.com",
    "https://developer.mozilla.org",
    "https://openai.com",
    "https://www.wikipedia.org",
    "https://example.com",
]

print("=" * 60)
print("SEEKER PIPELINE DIAGNOSIS")
print("=" * 60)

# ── Step 1: Test fast_reject ──────────────────────────────────────
print("\n[1] Fast-reject check (seeker.py):")
try:
    from discovery.sitemap_fetcher.seeker.seeker import _fast_reject
    for url in test_urls:
        rejected = _fast_reject(url)
        print(f"  {'REJECTED' if rejected else 'OK      '} — {url}")
except ImportError as e:
    print(f"  Could not import _fast_reject: {e}")
    print("  Trying manual check...")
    import re
    _SCHEME_RE = re.compile(r'^https?://', re.IGNORECASE)
    for url in test_urls:
        ok = bool(_SCHEME_RE.match(url))
        print(f"  {'OK      ' if ok else 'REJECTED'} — {url}")

# ── Step 2: Test url_parser ───────────────────────────────────────
print("\n[2] URL parser check:")
try:
    from discovery.sitemap_fetcher.seeker.url_parser import parse
    for url in test_urls:
        try:
            parser = parse(url)
            scheme = parser.find_scheme()
            host   = parser.host()
            path   = parser.path()
            print(f"  OK  scheme={scheme!r}  host={host!r}  path={path!r}  — {url}")
        except Exception as e:
            print(f"  FAIL — {url}: {e}")
except ImportError as e:
    print(f"  Could not import url_parser: {e}")

# ── Step 3: Test url_model ────────────────────────────────────────
print("\n[3] URL model check:")
try:
    from discovery.sitemap_fetcher.seeker.url_parser import parse
    from discovery.sitemap_fetcher.models.urls_model import url_model
    for url in test_urls:
        try:
            parser = parse(url)
            model  = url_model(
                scheme   = parser.find_scheme(),
                user     = parser.user(),
                password = parser.password(),
                host     = parser.host(),
                port     = parser.port(),
                path     = parser.path(),
                query    = parser.query_direct(),
                fragment = parser.fragment(),
            )
            print(f"  OK  — {url}")
        except Exception as e:
            print(f"  FAIL — {url}: {e}")
except ImportError as e:
    print(f"  Could not import url_model: {e}")

# ── Step 4: Test validator ────────────────────────────────────────
print("\n[4] Validator check:")
try:
    from discovery.sitemap_fetcher.seeker.url_parser import parse
    from discovery.sitemap_fetcher.models.urls_model import url_model
    from discovery.sitemap_fetcher.seeker.validator import Validate
    for url in test_urls:
        try:
            parser = parse(url)
            model  = url_model(
                scheme=parser.find_scheme(), user=parser.user(),
                password=parser.password(), host=parser.host(),
                port=parser.port(), path=parser.path(),
                query=parser.query_direct(), fragment=parser.fragment(),
            )
            v = Validate.validate(model)
            print(f"  {'VALID  ' if v.is_valid else 'INVALID'} errors={v.errors} — {url}")
        except Exception as e:
            print(f"  FAIL — {url}: {e}")
except ImportError as e:
    print(f"  Could not import Validate: {e}")

# ── Step 5: Test normalizer ───────────────────────────────────────
print("\n[5] Normalizer check:")
try:
    from discovery.sitemap_fetcher.seeker.url_parser import parse
    from discovery.sitemap_fetcher.models.urls_model import url_model
    from discovery.sitemap_fetcher.seeker.validator import Validate
    from discovery.sitemap_fetcher.seeker.normalizer import URLNormazer
    for url in test_urls:
        try:
            parser = parse(url)
            model  = url_model(
                scheme=parser.find_scheme(), user=parser.user(),
                password=parser.password(), host=parser.host(),
                port=parser.port(), path=parser.path(),
                query=parser.query_direct(), fragment=parser.fragment(),
            )
            v = Validate.validate(model)
            if not v.is_valid:
                print(f"  SKIP (invalid) — {url}")
                continue
            norm = URLNormazer.normalize(model)
            print(f"  OK  normalized={norm.normalized!r} — {url}")
        except Exception as e:
            print(f"  FAIL — {url}: {e}")
except ImportError as e:
    print(f"  Could not import URLNormazer: {e}")

# ── Step 6: Test full seekingProcessing.processes() ───────────────
print("\n[6] Full seekingProcessing.processes() check:")
try:
    from discovery.sitemap_fetcher.seeker.seeker import seekingProcessing
    for url in test_urls:
        try:
            result = seekingProcessing.processes(url)
            if result is None:
                print(f"  NONE (returned None) — {url}")
            else:
                norm = result.normalized if hasattr(result, 'normalized') else str(result)
                print(f"  OK  normalized={norm!r} — {url}")
        except Exception as e:
            print(f"  FAIL — {url}: {e}")
except ImportError as e:
    print(f"  Could not import seekingProcessing: {e}")

print("\n" + "=" * 60)
print("Check which step first shows FAIL/NONE/INVALID above.")
print("That is where the bug is.")
print("=" * 60)