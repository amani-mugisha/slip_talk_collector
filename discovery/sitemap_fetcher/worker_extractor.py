from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import redis.asyncio as aioredis

from discovery.sitemap_fetcher.Extractor.content_extractor import ContentExtractor


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("extractor_worker")


# ── Configuration ─────────────────────────────────────────────────────────────

REDIS_HOST          = os.getenv("REDIS_HOST",         "localhost")
REDIS_PORT          = int(os.getenv("REDIS_PORT",     "6379"))
REDIS_DB            = int(os.getenv("REDIS_DB",       "0"))
REDIS_PASSWORD      = os.getenv("REDIS_PASSWORD",     None)

INPUT_QUEUE         = os.getenv("INPUT_QUEUE",        "html_queue")
PROCESSING_QUEUE    = os.getenv("PROCESSING_QUEUE",   "extracting_queue")
OUTPUT_QUEUE        = os.getenv("OUTPUT_QUEUE",       "clean_text_queue")
DEAD_LETTER_QUEUE   = os.getenv("DEAD_LETTER_QUEUE",  "dead_letter_queue")

CONCURRENCY         = int(os.getenv("CONCURRENCY",       "200"))
BATCH_SIZE          = int(os.getenv("BATCH_SIZE",        "20"))
IDLE_SHUTDOWN_AFTER = int(os.getenv("IDLE_SHUTDOWN_AFTER","0"))   # 0 = never
THREAD_POOL_SIZE    = int(os.getenv("THREAD_POOL_SIZE",  "8"))

RECONNECT_BASE      = float(os.getenv("RECONNECT_BASE",  "1.0"))
RECONNECT_MAX       = float(os.getenv("RECONNECT_MAX",   "30.0"))


# ── Graceful shutdown ─────────────────────────────────────────────────────────

_shutdown = False

def _handle_signal(sig, frame):
    global _shutdown
    log.info("Signal %s received — finishing in-flight pages then shutting down.", sig)
    _shutdown = True

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)


# ── Stats ─────────────────────────────────────────────────────────────────────

class Stats:
    def __init__(self):
        self.processed   = 0
        self.skipped     = 0
        self.errors      = 0
        self.dead_letter = 0
        self.start_time  = time.perf_counter()
        self._lock       = asyncio.Lock()

    async def inc(self, processed=0, skipped=0, errors=0, dead=0):
        async with self._lock:
            self.processed   += processed
            self.skipped     += skipped
            self.errors      += errors
            self.dead_letter += dead

    def report(self) -> str:
        elapsed = time.perf_counter() - self.start_time
        rate    = self.processed / max(elapsed / 60, 0.001)
        return (
            f"processed={self.processed}  skipped={self.skipped}  "
            f"errors={self.errors}  dead_letter={self.dead_letter}  "
            f"rate={rate:.0f} pages/min  elapsed={elapsed:.0f}s"
        )


# ── Redis connection ────────────────────────────────────────

async def _connect_with_backoff() -> aioredis.Redis:
    delay   = RECONNECT_BASE
    attempt = 0
    while True:
        attempt += 1
        try:
            client = aioredis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                max_connections=CONCURRENCY + 20,
                socket_connect_timeout=5,
                socket_keepalive=True,
                decode_responses=False,
            )
            await client.ping()
            log.info(
                "Connected to Redis %s:%d db=%d (attempt %d)",
                REDIS_HOST, REDIS_PORT, REDIS_DB, attempt,
            )
            return client
        except Exception as exc:
            log.warning(
                "Redis connection failed (attempt %d): %s — retrying in %.1fs",
                attempt, exc, delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX)


# ── Batch URL popper ──────────────────────────────────────────────────────────

async def _pop_batch(
    client: aioredis.Redis,
    batch_size: int,
) -> list[bytes]:
    """
    Pop up to batch_size items from html_queue atomically using a pipeline.
    Moves each item to extracting_queue before returning.
    Much faster than individual BLMOVE calls.
    """
    # Use pipeline to pop + push to processing queue atomically
    pipeline = client.pipeline()
    for _ in range(batch_size):
        pipeline.rpoplpush(INPUT_QUEUE, PROCESSING_QUEUE)
    results = await pipeline.execute()
    return [r for r in results if r is not None]


# ── Message decoding ──────────────────────────────────────────────────────────

def _decode_raw(raw: bytes) -> Optional[str]:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def _parse_payload(raw_str: str) -> Optional[dict]:
    try:
        payload = json.loads(raw_str)
    except json.JSONDecodeError as exc:
        log.error("Invalid JSON: %s — snippet: %.120s", exc, raw_str)
        return None
    if not isinstance(payload, dict):
        return None
    if not payload.get("html"):
        log.warning("Payload missing 'html' — url=%s", payload.get("url", "?"))
        return None
    return payload


# ── Core extraction (runs in thread pool — CPU bound) ─────────────────────────

def _extract_sync(html: str, url: str) -> Optional[dict]:
    """
    Runs ContentExtractor.extract() synchronously.
    Called via loop.run_in_executor() to avoid blocking the event loop.
    """
    return ContentExtractor.extract(html, url=url)


# ── Per-message processor ─────────────────────────────────────────────────────

async def _process_one(
    raw: bytes,
    client: aioredis.Redis,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
    stats: Stats,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Process one HTML message end-to-end."""

    async with semaphore:
        raw_str = _decode_raw(raw)
        payload = _parse_payload(raw_str)

        if payload is None:
            # Dead letter — bad JSON or missing html field
            try:
                record = json.dumps({
                    "reason":    "invalid_json_or_missing_html",
                    "timestamp": time.time(),
                    "raw":       raw_str[:2000] if raw_str else "",
                })
                await client.rpush(DEAD_LETTER_QUEUE, record)
            except Exception:
                pass
            await stats.inc(dead=1, errors=1)
            return

        url  = payload.get("url", "")
        html = payload["html"]

        # ── Run CPU-bound extraction in thread pool ───────────────────────
        t0 = time.perf_counter()
        try:
            result = await loop.run_in_executor(executor, _extract_sync, html, url)
        except Exception:
            log.error(
                "ContentExtractor exception for url=%s:\n%s",
                url, traceback.format_exc(),
            )
            record = json.dumps({
                "reason":    "extractor_exception",
                "timestamp": time.time(),
                "url":       url,
            })
            await client.rpush(DEAD_LETTER_QUEUE, record)
            await stats.inc(dead=1, errors=1)
            return
        finally:
            # Always remove from processing queue
            try:
                await client.lrem(PROCESSING_QUEUE, 1, raw)
            except Exception as exc:
                log.warning("Failed to remove from processing queue: %s", exc)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if result is None:
            log.debug("No content extracted  url=%s  (%.0fms)", url, elapsed_ms)
            await stats.inc(skipped=1)
            return

        # ── Push clean text to output queue ──────────────────────────────
        await client.rpush(
            OUTPUT_QUEUE,
            json.dumps(result, ensure_ascii=False),
        )
        log.debug(
            "Extracted  url=%-60s  chars=%-6d  (%.0fms)",
            url, len(result["text"]), elapsed_ms,
        )
        await stats.inc(processed=1)


# ── Progress logger ───────────────────────────────────────────────────────────

async def _log_progress(stats: Stats, client: aioredis.Redis):
    while not _shutdown:
        await asyncio.sleep(10)
        try:
            html_q   = await client.llen(INPUT_QUEUE)
            clean_q  = await client.llen(OUTPUT_QUEUE)
            proc_q   = await client.llen(PROCESSING_QUEUE)
            log.info(
                "%s  |  html_queue=%d  extracting=%d  clean_text=%d",
                stats.report(), html_q, proc_q, clean_q,
            )
        except Exception:
            pass


# ── Main async loop ───────────────────────────────────────────────────────────

async def main():
    log.info("=" * 60)
    log.info("Extractor Worker starting  (async high-throughput mode)")
    log.info("  Concurrency    : %d coroutines", CONCURRENCY)
    log.info("  Batch size     : %d pages per pop", BATCH_SIZE)
    log.info("  Thread pool    : %d threads (for ContentExtractor)", THREAD_POOL_SIZE)
    log.info("  Input queue    : %s", INPUT_QUEUE)
    log.info("  Output queue   : %s", OUTPUT_QUEUE)
    log.info("=" * 60)

    client    = await _connect_with_backoff()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    executor  = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)
    stats     = Stats()
    loop      = asyncio.get_event_loop()
    tasks: set[asyncio.Task] = set()

    # Start progress logger
    asyncio.create_task(_log_progress(stats, client))

    idle_elapsed = 0
    log.info("Waiting for pages in '%s'...", INPUT_QUEUE)

    try:
        while not _shutdown:
            # ── Pop a batch from html_queue ───────────────────────────────
            try:
                batch = await _pop_batch(client, BATCH_SIZE)
            except aioredis.RedisError as exc:
                log.warning("Redis error: %s — reconnecting...", exc)
                client = await _connect_with_backoff()
                continue

            if not batch:
                # Queue empty
                await asyncio.sleep(1)
                idle_elapsed += 1

                if IDLE_SHUTDOWN_AFTER and idle_elapsed >= IDLE_SHUTDOWN_AFTER:
                    log.info(
                        "Idle for %ds — shutting down.", IDLE_SHUTDOWN_AFTER
                    )
                    break

                if idle_elapsed % 30 == 0:
                    log.info("Waiting for pages... idle=%ds", idle_elapsed)
                continue

            idle_elapsed = 0

            # ── Dispatch extraction coroutines ────────────────────────────
            for raw in batch:
                if _shutdown:
                    break
                task = asyncio.create_task(
                    _process_one(raw, client, semaphore, executor, stats, loop)
                )
                tasks.add(task)
                task.add_done_callback(tasks.discard)

            # ── Backpressure — don't pile up more than 2× CONCURRENCY ─────
            while len(tasks) > CONCURRENCY * 2 and not _shutdown:
                await asyncio.sleep(0.05)

    finally:
        log.info("Waiting for %d in-flight extractions to finish...", len(tasks))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        executor.shutdown(wait=True)
        await client.aclose()
        log.info("Final stats: %s", stats.report())
        log.info("Extractor worker stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(main())
