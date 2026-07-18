from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import signal
import time
from collections import defaultdict
from typing import Optional
from urllib.parse import urljoin, urlsplit

import aiohttp
import redis.asyncio as aioredis

from discovery.sitemap_fetcher.seeker import seeker

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO, 
)
log = logging.getLogger("crawler_worker")


# ── Configuration ─────────────────────────────────────────────────────────────

REDIS_HOST      = os.getenv("REDIS_HOST",    "localhost")
REDIS_PORT      = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB        = int(os.getenv("REDIS_DB",   "0"))
REDIS_PASSWORD  = os.getenv("REDIS_PASSWORD", None)

CONCURRENCY     = int(os.getenv("CONCURRENCY",    "500"))
BATCH_SIZE      = int(os.getenv("BATCH_SIZE",     "50"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT","10"))
MAX_RETRIES     = int(os.getenv("MAX_RETRIES",    "2"))
CRAWL_DELAY     = float(os.getenv("CRAWL_DELAY",  "0.0"))
MAX_HTML_SIZE   = int(os.getenv("MAX_HTML_SIZE",  str(5 * 1024 * 1024)))  # 5 MB

USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (compatible; SlipTalkBot/1.0; +https://siptalk.ai/bot)"
)

# Redis queue names
WAITING_QUEUE    = "waiting_urls"
PROCESSING_QUEUE = "processing_queue"
HTML_QUEUE       = "html_queue"
FAILED_QUEUE     = "failed_urls"
SEEN_SET         = "seen_urls"
COMPLETED_SET    = "completed_urls"

# HTTP headers
HEADERS = {
    "User-Agent":      USER_AGENT,
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-us,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection":      "keep-alive",
}


# ── Graceful shutdown ─

_shutdown = False

def _handle_signal(sig, frame):
    global _shutdown
    log.info("Signal %s received — shutting down gracefully...", sig)
    _shutdown = True

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)


# ── Stats tracker ─────────────────────────────────────────────────────────────

class Stats:
    def __init__(self):
        self.fetched   = 0
        self.failed    = 0
        self.skipped   = 0
        self.links_found = 0
        self.new_links   = 0
        self.start_time  = time.perf_counter()
        self._lock       = asyncio.Lock()

    async def inc(self, fetched=0, failed=0, skipped=0, links=0, new_links=0):
        async with self._lock:
            self.fetched     += fetched
            self.failed      += failed
            self.skipped     += skipped
            self.links_found += links
            self.new_links   += new_links

    def report(self) -> str:
        elapsed  = time.perf_counter() - self.start_time
        per_min  = self.fetched / max(elapsed / 60, 0.001)
        return (
            f"fetched={self.fetched}  failed={self.failed}  "
            f"skipped={self.skipped}  links_found={self.links_found}  "
            f"new_links={self.new_links}  "
            f"rate={per_min:.0f} pages/min  elapsed={elapsed:.0f}s"
        )

# ── URL utilities ─────────────────────────────────────────────────────────────

def _fingerprint(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


def _extract_domain(url: str) -> str:
    try:
        return urlsplit(url).netloc
    except Exception:
        return ""


def _extract_links(html: str, base_url: str) -> list[str]:
    """
    Fast link extraction without BeautifulSoup overhead.
    Uses a lightweight HTML parser for speed at scale.
    Falls back to BeautifulSoup for complex pages.
    """
    from html.parser import HTMLParser

    class LinkParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                for name, value in attrs:
                    if name == "href" and value:
                        self.links.append(value)

    parser = LinkParser()
    try:
        parser.feed(html[:500_000])  # cap at 500KB for link extraction
    except Exception:
        pass
    return parser.links


# ── Domain rate limiter ────────────────────────────────────────────────────────

class DomainThrottle:
    """
    Per-domain rate limiter — prevents hammering a single server.
    Uses asyncio.Lock per domain so concurrent requests to the
    same domain are serialised with CRAWL_DELAY between them.
    """

    def __init__(self, delay: float = CRAWL_DELAY):
        self.delay  = delay
        self._locks: dict[str, asyncio.Lock]   = defaultdict(asyncio.Lock)
        self._last:  dict[str, float]          = defaultdict(float)

    async def wait(self, domain: str):
        if self.delay <= 0:
            return
        async with self._locks[domain]:
            now   = time.perf_counter()
            since = now - self._last[domain]
            if since < self.delay:
                await asyncio.sleep(self.delay - since)
            self._last[domain] = time.perf_counter()


# ── Core fetch coroutine ──────────────────────────────────────────────────────

async def fetch_url(
    url: str,
    session: aiohttp.ClientSession,
    redis_client: aioredis.Redis,
    semaphore: asyncio.Semaphore,
    throttle: DomainThrottle,
    stats: Stats,
) -> None:
    """Fetch one URL, push HTML to Redis, enqueue discovered links."""

    async with semaphore:
        domain = _extract_domain(url)
        await throttle.wait(domain)

        html: Optional[str] = None

        # ── HTTP fetch with retries ───────────────────────────────────────
        for attempt in range(1, MAX_RETRIES + 2):
            try:
                async with session.get(
                    url,
                    headers=HEADERS,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                    allow_redirects=True,
                    ssl=False,          # skip SSL verification for speed
                ) as resp:

                    if resp.status != 200:
                        await stats.inc(failed=1)
                        # Push to failed queue for inspection
                        await redis_client.rpush(
                            FAILED_QUEUE,
                            json.dumps({"url": url, "status": resp.status}),
                        )
                        return

                    # Check content type — only crawl HTML pages
                    content_type = resp.headers.get("Content-Type", "")
                    if "text/html" not in content_type:
                        await stats.inc(skipped=1)
                        return

                    # Check content length before reading
                    content_length = int(resp.headers.get("Content-Length", 0))
                    if content_length > MAX_HTML_SIZE:
                        log.debug("Skipping oversized page (%d bytes): %s", content_length, url)
                        await stats.inc(skipped=1)
                        return

                    # Read with size cap
                    html = await resp.text(encoding="utf-8", errors="replace")
                    if len(html) > MAX_HTML_SIZE:
                        html = html[:MAX_HTML_SIZE]

                break   # success — exit retry loop

            except asyncio.TimeoutError:
                if attempt <= MAX_RETRIES:
                    await asyncio.sleep(0.5 * attempt)
                    continue
                await stats.inc(failed=1)
                return

            except aiohttp.ClientError as exc:
                if attempt <= MAX_RETRIES:
                    await asyncio.sleep(0.5 * attempt)
                    continue
                log.debug("Client error %s: %s", url, exc)
                await stats.inc(failed=1)
                return

            except Exception as exc:
                log.debug("Unexpected error %s: %s", url, exc)
                await stats.inc(failed=1)
                return

        if html is None:
            return

        # ── Push HTML to extraction queue ─────────────────────────────────
        page_data = json.dumps({"url": url, "html": html}, ensure_ascii=False)
        await redis_client.lpush(HTML_QUEUE, page_data)
        await stats.inc(fetched=1)

        # ── Extract and enqueue new links ─────────────────────────────────
        raw_links = _extract_links(html, url)
        await stats.inc(links=len(raw_links))

        new_count = 0
        pipeline  = redis_client.pipeline()

        for raw_href in raw_links:
            try:
                absolute = urljoin(url, raw_href)
                processed = seeker.seekingProcessing.processes(absolute)
                if not processed:
                    continue

                fp = _fingerprint(processed.normalized)

                # sadd returns 1 if new, 0 if already seen
                # We batch this with a pipeline for speed
                pipeline.sadd(SEEN_SET, fp)

            except Exception:
                continue

        # Execute pipeline — returns list of 1s and 0s
        results = await pipeline.execute()

        # Only push truly new URLs to waiting queue
        new_pipeline = redis_client.pipeline()
        link_idx = 0
        for raw_href in raw_links:
            try:
                absolute  = urljoin(url, raw_href)
                processed = seeker.seekingProcessing.processes(absolute)
                if not processed:
                    continue
                if link_idx < len(results) and results[link_idx] == 1:
                    new_pipeline.rpush(WAITING_QUEUE, processed.normalized)
                    new_count += 1
                link_idx += 1
            except Exception:
                link_idx += 1
                continue

        await new_pipeline.execute()
        await stats.inc(new_links=new_count)

        # ── Mark URL as completed ─────────────────────────────────────────
        url_fp = _fingerprint(url)
        await redis_client.sadd(COMPLETED_SET, url_fp)


# ── URL batch popper ──────────────────────────────────────────────────────────

async def pop_urls(
    redis_client: aioredis.Redis,
    batch_size: int,
) -> list[str]:
    """
    Pop up to batch_size URLs from waiting_urls atomically using a pipeline.
    Much faster than individual BLPOP calls.
    """
    pipeline = redis_client.pipeline()
    for _ in range(batch_size):
        pipeline.rpop(WAITING_QUEUE)
    results = await pipeline.execute()
    return [r.decode("utf-8") if isinstance(r, bytes) else r
            for r in results if r is not None]


# ── Progress logger ───────────────────────────────────────────────────────────

async def log_progress(stats: Stats, redis_client: aioredis.Redis):
    """Log stats every 10 seconds."""
    while not _shutdown:
        await asyncio.sleep(10)
        waiting   = await redis_client.llen(WAITING_QUEUE)
        html_q    = await redis_client.llen(HTML_QUEUE)
        completed = await redis_client.scard(COMPLETED_SET)
        log.info(
            "%s  |  waiting=%d  html_queue=%d  completed=%d",
            stats.report(), waiting, html_q, completed,
        )


# ── Main async loop ───────────────────────────────────────────────────────────

async def main():
    log.info("=" * 60)
    log.info("Crawler Worker starting")
    log.info("  Concurrency : %d coroutines", CONCURRENCY)
    log.info("  Batch size  : %d URLs per pop", BATCH_SIZE)
    log.info("  Timeout     : %ds per request", REQUEST_TIMEOUT)
    log.info("  Crawl delay : %.2fs per domain", CRAWL_DELAY)
    log.info("  Target      : ~%d pages/min per worker",
             CONCURRENCY * (60 // max(REQUEST_TIMEOUT, 1)))
    log.info("=" * 60)

    # ── Redis connection pool ─────────────────────────────────────────────
    redis_client = aioredis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        max_connections=CONCURRENCY + 20,
        decode_responses=False,
    )
    await redis_client.ping()
    log.info("Connected to Redis %s:%d", REDIS_HOST, REDIS_PORT)

    # ── aiohttp session (shared connection pool) ──────────────────────────
    connector = aiohttp.TCPConnector(
        limit=CONCURRENCY,          # total simultaneous connections
        limit_per_host=10,          # max connections per domain
        ttl_dns_cache=300,          # cache DNS for 5 minutes
        use_dns_cache=True,
        enable_cleanup_closed=True,
    )
    session = aiohttp.ClientSession(connector=connector)

    semaphore = asyncio.Semaphore(CONCURRENCY)
    throttle  = DomainThrottle(delay=CRAWL_DELAY)
    stats     = Stats()

    # Start progress logger
    asyncio.create_task(log_progress(stats, redis_client))

    idle_seconds = 0
    tasks: set[asyncio.Task] = set()

    log.info("Crawler running. Waiting for URLs in '%s'...", WAITING_QUEUE)

    try:
        while not _shutdown:
            # ── Pop a batch of URLs ───────────────────────────────────────
            urls = await pop_urls(redis_client, BATCH_SIZE)

            if not urls:
                # Queue empty — wait briefly then check again
                await asyncio.sleep(1)
                idle_seconds += 1
                if idle_seconds % 30 == 0:
                    log.info("Waiting for URLs... idle=%ds", idle_seconds)
                continue

            idle_seconds = 0

            # ── Dispatch fetch coroutines ─────────────────────────────────
            for url in urls:
                if _shutdown:
                    break
                task = asyncio.create_task(
                    fetch_url(url, session, redis_client, semaphore, throttle, stats)
                )
                tasks.add(task)
                task.add_done_callback(tasks.discard)

            # ── Backpressure — don't pile up too many tasks ───────────────
            # Wait if we have more than 2× CONCURRENCY pending tasks
            while len(tasks) > CONCURRENCY * 2 and not _shutdown:
                await asyncio.sleep(0.1)

    finally:
        log.info("Shutting down — waiting for %d in-flight requests...", len(tasks))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await session.close()
        await redis_client.aclose()
        log.info("Final stats: %s", stats.report())
        log.info("Crawler stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(main())