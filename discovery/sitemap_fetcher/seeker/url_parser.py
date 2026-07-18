import re
from urllib.parse import urlsplit, parse_qs, unquote, quote

from discovery.sitemap_fetcher.models.urls_model import url_model  # <-- adjust to your project's import path


# ---------------------------------------------------------------------------
# Well-known default ports, used to drop redundant ":80"/":443" etc. during
# normalization (RFC 3986 6.2.3 — "the port syntax component is omitted if
# it matches the default").
# ---------------------------------------------------------------------------
_DEFAULT_PORTS = {
    'http': 80, 'https': 443,
    'ftp': 21, 'ftps': 990,
    'ws': 80, 'wss': 443,
}

# A short, deliberately incomplete list of multi-label public suffixes.
# Real TLD splitting needs the full Public Suffix List (the `tldextract`
# package wraps it and stays up to date); this is a best-effort fallback
# so subdomain/domain/tld extraction doesn't need a network fetch or a
# vendored multi-thousand-line file. Swap in `tldextract` if precision
# here matters for your model's features.
_MULTI_LABEL_TLDS = {
    'co.uk', 'org.uk', 'ac.uk', 'gov.uk',
    'co.jp', 'ne.jp', 'or.jp',
    'com.au', 'net.au', 'org.au',
    'com.br', 'com.cn', 'com.mx',
    'co.in', 'co.nz', 'co.za',
}


def _remove_dot_segments(path):
    """RFC 3986 5.2.4 — collapse '/a/../b' -> '/b', './a' -> 'a', etc."""
    inp = path
    out = []
    while inp:
        if inp.startswith('../'):
            inp = inp[3:]
        elif inp.startswith('./'):
            inp = inp[2:]
        elif inp.startswith('/./'):
            inp = '/' + inp[3:]
        elif inp == '/.':
            inp = '/'
        elif inp.startswith('/../'):
            inp = '/' + inp[4:]
            if out:
                out.pop()
        elif inp == '/..':
            inp = '/'
            if out:
                out.pop()
        elif inp in ('.', '..'):
            inp = ''
        else:
            if inp.startswith('/'):
                rest = inp[1:]
                seg, sep, rest2 = rest.partition('/')
                out.append('/' + seg)
                inp = sep + rest2
            else:
                seg, sep, rest2 = inp.partition('/')
                out.append(seg)
                inp = sep + rest2
    return ''.join(out)


def _normalize_pct_encoding(s):
    """Re-encode so percent-escapes use uppercase hex (RFC 3986 6.2.2.1)
    and unreserved characters that were escaped unnecessarily get decoded,
    e.g. '%7E' -> '~'. Leaves reserved/unsafe characters escaped."""
    if not s:
        return s
    return quote(unquote(s, errors='replace'), safe="/:@!$&'()*+,;=~")


def normalize(scheme, host, port, path, query, fragment, drop_fragment=False):
    """Build a canonical URL string for dedup/embedding purposes.

    Steps applied (all from RFC 3986 section 6.2, "safe" normalizations
    that don't change what resource the URL refers to):
      1. lowercase scheme and host
      2. drop the port if it's the scheme's default
      3. resolve dot-segments in the path, default to '/'
      4. normalize percent-encoding case/unnecessary escapes
      5. leave query as-is (sorting would change semantics for APIs that
         are order-sensitive, so it's left to the caller)

    `drop_fragment` is off by default since fragment removal isn't a
    strictly meaning-preserving normalization in general, but sitemap-style
    dedup often wants it — pass True if your crawler treats
    "/page#section" and "/page" as the same document.
    """
    scheme = (scheme or 'https').lower()
    host = (host or '').lower().rstrip('.')  # trailing dot in DNS names is equivalent

    # IPv6 literals must be re-bracketed, or the ':' in the address is
    # indistinguishable from a port separator (urlsplit's .hostname
    # already strips the brackets off when it parses these).
    display_host = f"[{host}]" if ':' in host else host

    netloc = display_host
    if port and port != _DEFAULT_PORTS.get(scheme):
        netloc = f"{display_host}:{port}"

    norm_path = _normalize_pct_encoding(_remove_dot_segments(path or '/')) or '/'
    norm_query = _normalize_pct_encoding(query) if query else ''
    norm_fragment = '' if drop_fragment else (fragment or '')

    result = f"{scheme}://{netloc}{norm_path}"
    if norm_query:
        result += f"?{norm_query}"
    if norm_fragment:
        result += f"#{norm_fragment}"
    return result


def parse_query_params(query):
    """Query string -> dict of lists, percent-decoded.
    Ready to feed `url_model.query_params` once that field is enabled."""
    if not query:
        return {}
    return parse_qs(query, keep_blank_values=True)


def extract_domain_parts(host):
    """Best-effort (subdomain, domain, tld) split. See _MULTI_LABEL_TLDS
    docstring above for the accuracy caveat — use `tldextract` instead if
    this needs to be correct against the full public suffix list.
    Ready to feed `url_model.subdomain` / `.domain` / `.tld` once those
    fields are enabled."""
    if not host or re.match(r'^\d{1,3}(\.\d{1,3}){3}$', host) or ':' in host:
        # IPv4 or IPv6 literal — no meaningful subdomain/domain/tld split
        return '', host, ''

    labels = host.split('.')
    if len(labels) < 2:
        return '', host, ''

    last_two = '.'.join(labels[-2:])
    if last_two in _MULTI_LABEL_TLDS and len(labels) >= 3:
        domain = labels[-3]
        subdomain = '.'.join(labels[:-3])
        return subdomain, domain, last_two

    tld = labels[-1]
    domain = labels[-2]
    subdomain = '.'.join(labels[:-2])
    return subdomain, domain, tld


def parse(url, drop_fragment=False):
    """Parse a raw URL string into a `url_model`.

    Scheme-less input (e.g. "example.com/path") is recovered by retrying
    with a synthetic '//' prefix, matching how browsers interpret
    address-bar input — sitemaps should always contain absolute URLs, but
    this keeps behavior sane if a relative/malformed one slips through.
    """
    raw = url or ""
    split = urlsplit(raw)

    if not split.scheme and not split.netloc and raw and not raw.startswith(('/', '#', '?')):
        guess = urlsplit('//' + raw)
        if guess.netloc:
            split = guess

    scheme = split.scheme or 'https'
    user = unquote(split.username) if split.username else ''
    password = unquote(split.password) if split.password else ''
    host = split.hostname or ''
    port = split.port  # int or None; url_model defaults port=None
    path = split.path or '/'
    query = split.query
    fragment = split.fragment

    normalized = normalize(scheme, host, port, path, query, fragment,
                            drop_fragment=drop_fragment)

    # subdomain, domain, tld = extract_domain_parts(host)   # once enabled on url_model
    # query_params = parse_query_params(query)               # once enabled on url_model

    return url_model(
        scheme=scheme,
        user=user,
        password=password,
        host=host,
        port=port,
        path=path,
        query=query,
        fragment=fragment,
        raw=raw,
        normalized=normalized,
        # subdomain=subdomain,
        # domain=domain,
        # tld=tld,
        # query_params=query_params,
    )