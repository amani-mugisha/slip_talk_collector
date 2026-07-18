"""
main_seeker.py
==============
High-performance seed URL seeder — feeds waiting_urls in bulk batches.

Problems with the original
---------------------------
1. URLGenerate() called one URL at a time — one Redis round-trip per seed
2. Synchronous — blocks on every Redis call
3. print() on every URL — I/O bottleneck at scale
4. Never used Generate.batch_generate() we built
5. Redis client created inside MainSeeker but also inside Generate
   — two separate connections for one job
6. No reconnect logic — Redis drop = silent crash
7. No throughput reporting — no visibility into actual speed

Run
---
    python3 -m discovery.sitemap_fetcher.main_seeker

Environment variables
----------------------
    REDIS_HOST          default: localhost
    REDIS_PORT          default: 6379
    REDIS_DB            default: 0
    BATCH_SIZE          default: 500
    LOG_EVERY           default: 1000  (log progress every N URLs pushed)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time

import redis.asyncio as aioredis

from discovery.sitemap_fetcher.seeker.generator import Generate


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("main_seeker")


# ── Configuration ─────────────────────────────────────────────────────────────

REDIS_HOST  = os.getenv("REDIS_HOST",   "localhost")
REDIS_PORT  = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB    = int(os.getenv("REDIS_DB",   "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
BATCH_SIZE  = int(os.getenv("BATCH_SIZE",  "500"))
LOG_EVERY   = int(os.getenv("LOG_EVERY",   "1000"))


# ── Graceful shutdown ─────────────────────────────────────────────────────────

_shutdown = False

def _handle_signal(sig, frame):
    global _shutdown
    log.info("Signal %s — finishing current batch then stopping.", sig)
    _shutdown = True

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)


# ── Redis connection ──────────────────────────────────────────────────────────

async def _connect() -> aioredis.Redis:
    delay = 1.0
    attempt = 0
    while True:
        attempt += 1
        try:
            client = aioredis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_keepalive=True,
            )
            await client.ping()
            log.info(
                "Connected to Redis %s:%d db=%d (attempt %d)",
                REDIS_HOST, REDIS_PORT, REDIS_DB, attempt,
            )
            return client
        except Exception as exc:
            log.warning(
                "Redis connection failed (attempt %d): %s — retry in %.1fs",
                attempt, exc, delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)


# ════════════════════════════════════════════════════════════════════════════════
# MAIN SEEKER
# ════════════════════════════════════════════════════════════════════════════════

class MainSeeker:

    @staticmethod
    async def run_async():
        log.info("=" * 56)
        log.info("Main Seeker starting")
        log.info("  Batch size : %d URLs per round-trip", BATCH_SIZE)
        log.info("  Log every  : %d URLs pushed", LOG_EVERY)
        log.info("=" * 56)

        client = await _connect()

        total_popped  = 0
        total_pushed  = 0
        total_skipped = 0
        t0            = time.perf_counter()

        while not _shutdown:

            # ── Check if seeds queue still has items ──────────────────────
            try:
                remaining = await client.llen("seeds_urls_queue")
            except aioredis.RedisError as exc:
                log.warning("Redis error: %s — reconnecting...", exc)
                client = await _connect()
                continue

            if remaining == 0:
                log.info(
                    "All seed URLs processed. "
                    "pushed=%d  skipped=%d  elapsed=%.1fs",
                    total_pushed, total_skipped,
                    time.perf_counter() - t0,
                )
                break

            # ── Batch generate: pop N seeds, process, push to waiting_urls
            try:
                result = await Generate.batch_generate(
                    client,
                    batch_size=min(BATCH_SIZE, remaining),
                )
            except aioredis.RedisError as exc:
                log.warning("Redis error during batch: %s — reconnecting...", exc)
                client = await _connect()
                continue

            total_popped  += result["popped"]
            total_pushed  += result["pushed"]
            total_skipped += result["skipped"]

            # ── Progress log every LOG_EVERY URLs ─────────────────────────
            if total_pushed > 0 and total_pushed % LOG_EVERY < result["pushed"]:
                elapsed = time.perf_counter() - t0
                rate    = total_pushed / max(elapsed / 60, 0.001)
                log.info(
                    "pushed=%-6d  skipped=%-5d  seeds_left=%-6d  "
                    "rate=%.0f URLs/min",
                    total_pushed, total_skipped,
                    remaining - result["popped"],
                    rate,
                )

        await client.aclose()

        elapsed = time.perf_counter() - t0
        log.info("=" * 56)
        log.info("Main Seeker finished")
        log.info("  Total popped  : %d", total_popped)
        log.info("  Total pushed  : %d", total_pushed)
        log.info("  Total skipped : %d", total_skipped)
        log.info("  Elapsed       : %.2fs", elapsed)
        log.info(
            "  Throughput    : %.0f URLs/min",
            total_pushed / max(elapsed / 60, 0.001),
        )
        log.info("=" * 56)

    @staticmethod
    def run():
        """Sync entry point — wraps async run."""
        asyncio.run(MainSeeker.run_async())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MainSeeker.run()