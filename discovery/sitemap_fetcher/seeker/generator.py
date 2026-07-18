from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import signal
import time
from typing import Optional

import redis.asyncio as aioredis

from discovery.sitemap_fetcher.seeker.seeker import seekingProcessing as sp


# ── Logging ───────

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("generator")


# ── Configuration

REDIS_HOST    = os.getenv("REDIS_HOST",    "localhost")
REDIS_PORT    = int(os.getenv("REDIS_PORT","6379"))
REDIS_DB      = int(os.getenv("REDIS_DB",  "0"))
REDIS_PASSWORD= os.getenv("REDIS_PASSWORD", None)

BATCH_SIZE    = int(os.getenv("BATCH_SIZE",  "500"))
SEEDS_QUEUE   = os.getenv("SEEDS_QUEUE",   "seeds_urls_queue")
WAITING_QUEUE = os.getenv("WAITING_QUEUE", "waiting_urls")
SEEN_SET      = os.getenv("SEEN_SET",      "seen_urls")


# ── Graceful shutdown ────

_shutdown = False

def _handle_signal(sig, frame):
    global _shutdown
    log.info("Signal %s — shutting down after current batch.", sig)
    _shutdown = True

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)


# ── Fingerprint ───────────────────────────────────────────────────────────────

def _fp(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


# ════════════════════════════════════════════════════════════════════════════════
# GENERATE CLASS  (keeps original interface + adds batch method)
# ════════════════════════════════════════════════════════════════════════════════

class Generate:
    """
    Seed URL generator.

    Two modes
    ---------
    Generate.URLGenerate()              — original single-URL interface (sync)
    await Generate.batch_generate(r)    — new high-throughput batch mode (async)
    """

    # ── Original interface (kept for backward compatibility) ──────────────────

    @staticmethod
    def URLGenerate(r_sync=None):
        """
        Original single-URL interface — unchanged behaviour.
        Uses a sync Redis client if provided, otherwise creates one.
        """
        import redis as redis_sync

        client = r_sync or redis_sync.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
            password=REDIS_PASSWORD,
        )

        raw = client.rpop(SEEDS_QUEUE)
        if raw is None:
            return None

        raw_url = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        processed = sp.processes(raw_url)
        if processed is None:
            return None

        url = processed.normalized if hasattr(processed, "normalized") else str(processed)
        client.lpush(WAITING_QUEUE, url)
        return processed

    # ── New high-throughput batch method ──────────────────────────────────────

    @staticmethod
    async def batch_generate(
        client: aioredis.Redis,
        batch_size: int = BATCH_SIZE,
    ) -> dict:
        """
        Pop up to batch_size seeds, process, dedup, and push to waiting_urls
        in two Redis round-trips (one pipeline pop + one pipeline push).

        Returns
        -------
        dict with keys: popped, valid, skipped, pushed
        """
        # ── Step 1: batch pop seeds (one pipeline round-trip) ────────────
        pop_pipeline = client.pipeline()
        for _ in range(batch_size):
            pop_pipeline.rpop(SEEDS_QUEUE)
        raw_results = await pop_pipeline.execute()

        # Filter out None (empty queue slots)
        raw_urls = [
            r.decode("utf-8") if isinstance(r, bytes) else r
            for r in raw_results
            if r is not None
        ]

        if not raw_urls:
            return {"popped": 0, "valid": 0, "skipped": 0, "pushed": 0}

        popped = len(raw_urls)

        # ── Step 2: process URLs through seeker pipeline ──────────────────
        valid_urls: list[str] = []
        skipped = 0

        for raw_url in raw_urls:
            try:
                processed = sp.processes(raw_url)
                if processed is None:
                    skipped += 1
                    continue
                url = processed.normalized if hasattr(processed, "normalized") else str(processed)
                if url:
                    valid_urls.append(url)
                else:
                    skipped += 1
            except Exception as exc:
                log.debug("Failed to process URL '%s': %s", raw_url, exc)
                skipped += 1

        if not valid_urls:
            return {"popped": popped, "valid": 0, "skipped": skipped, "pushed": 0}

        # ── Step 3: dedup check (one pipeline sadd round-trip) ───────────
        dedup_pipeline = client.pipeline()
        for url in valid_urls:
            dedup_pipeline.sadd(SEEN_SET, _fp(url))
        dedup_results = await dedup_pipeline.execute()

        # sadd returns 1 = new, 0 = already seen
        new_urls = [
            url for url, is_new in zip(valid_urls, dedup_results)
            if is_new == 1
        ]
        skipped += len(valid_urls) - len(new_urls)

        if not new_urls:
            return {
                "popped": popped, "valid": len(valid_urls),
                "skipped": skipped, "pushed": 0,
            }

        # ── Step 4: batch push to waiting_urls (one pipeline round-trip) ──
        push_pipeline = client.pipeline()
        for url in new_urls:
            push_pipeline.lpush(WAITING_QUEUE, url)
        await push_pipeline.execute()

        return {
            "popped":  popped,
            "valid":   len(valid_urls),
            "skipped": skipped,
            "pushed":  len(new_urls),
        }


# ════════════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ════════════════════════════════════════════════════════════════════════════════

async def main():
    log.info("=" * 56)
    log.info("Seed URL Generator starting")
    log.info("  Seeds queue   : %s", SEEDS_QUEUE)
    log.info("  Waiting queue : %s", WAITING_QUEUE)
    log.info("  Batch size    : %d", BATCH_SIZE)
    log.info("=" * 56)

    client = aioredis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=False,
    )
    await client.ping()
    log.info("Connected to Redis %s:%d", REDIS_HOST, REDIS_PORT)

    total_pushed = 0
    total_popped = 0
    t0           = time.perf_counter()
    idle_seconds = 0

    while not _shutdown:
        seeds_remaining = await client.llen(SEEDS_QUEUE)

        if seeds_remaining == 0:
            idle_seconds += 1
            if idle_seconds % 10 == 0:
                log.info("Seeds queue empty — waiting... (%ds idle)", idle_seconds)
            await asyncio.sleep(1)
            continue

        idle_seconds = 0

        # Process one batch
        result = await Generate.batch_generate(client, batch_size=BATCH_SIZE)

        total_popped += result["popped"]
        total_pushed += result["pushed"]

        elapsed = time.perf_counter() - t0
        rate    = total_pushed / max(elapsed / 60, 0.001)

        log.info(
            "Batch done — popped=%-4d  valid=%-4d  skipped=%-4d  pushed=%-4d  "
            "total_pushed=%-6d  rate=%.0f URLs/min  seeds_left=%d",
            result["popped"], result["valid"], result["skipped"], result["pushed"],
            total_pushed, rate, seeds_remaining - result["popped"],
        )

    await client.aclose()
    log.info(
        "Generator stopped. total_popped=%d  total_pushed=%d",
        total_popped, total_pushed,
    )


if __name__ == "__main__":
    asyncio.run(main())