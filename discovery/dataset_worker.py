import hashlib
import json
import logging
import os
import signal
import time
from typing import Iterator, Optional

import redis

from discovery.build_datasets.dataset_builder import DatasetBuilder, Document

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("dataset_worker")


# ── Configuration ─────────────────────────────────────────────────────────────

REDIS_HOST       = os.getenv("REDIS_HOST",        "localhost")
REDIS_PORT       = int(os.getenv("REDIS_PORT",    "6379"))
REDIS_DB         = int(os.getenv("REDIS_DB",      "0"))
REDIS_PASSWORD   = os.getenv("REDIS_PASSWORD",    None)
INPUT_QUEUE      = os.getenv("INPUT_QUEUE",       "clean_text_queue")
OUTPUT_DIR       = os.getenv("OUTPUT_DIR",        "dataset_output")
IDLE_SHUTDOWN    = int(os.getenv("IDLE_SHUTDOWN", "60"))
REDIS_BATCH_SIZE = int(os.getenv("REDIS_BATCH_SIZE", "500"))
SHARD_SIZE       = int(os.getenv("SHARD_SIZE",    "5000"))


# ── Graceful shutdown ─────────────────────────────────────────────────────────

_shutdown = False

def _handle_signal(sig, frame):
    global _shutdown
    log.info("Signal %s received — flushing buffer and shutting down.", sig)
    _shutdown = True

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)


# ── Redis connection ──────────────────────────────────────────────────────────

def _make_client() -> redis.Redis:
    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_keepalive=True,
        retry_on_timeout=True,
    )
    client.ping()
    return client


def _connect_with_backoff() -> redis.Redis:
    delay = 1.0
    attempt = 0
    while True:
        attempt += 1
        try:
            client = _make_client()
            log.info(
                "Connected to Redis %s:%d db=%d queue=%s (attempt %d)",
                REDIS_HOST, REDIS_PORT, REDIS_DB, INPUT_QUEUE, attempt,
            )
            return client
        except Exception as exc:
            log.warning("Redis connection failed (attempt %d): %s — retry in %.1fs", attempt, exc, delay)
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


# ── Batch document stream ─────────────────────────────────────────────────────

def _batch_stream(
    client: redis.Redis,
    queue_name: str,
    batch_size: int,
    idle_shutdown_after: int,
) -> Iterator[Document]:
    """
    Yield Documents by popping Redis in batches.

    One pipeline round-trip pops `batch_size` documents at once —
    vs the original blpop() which did one round-trip per document.

    At batch_size=500:
      Original : 500 round-trips to get 500 docs
      New      : 1 round-trip to get 500 docs
    """
    global _shutdown
    idle_elapsed = 0

    while not _shutdown:

        # ── Batch pop: one pipeline call pops N docs ──────────────────────
        try:
            pipeline = client.pipeline()
            for _ in range(batch_size):
                pipeline.rpop(queue_name)
            results = pipeline.execute()
        except redis.RedisError as exc:
            log.warning("Redis error: %s — reconnecting...", exc)
            time.sleep(3)
            try:
                client = _connect_with_backoff()
            except Exception:
                pass
            continue

        # Filter None (empty slots at end of pipeline)
        raw_docs = [r for r in results if r is not None]

        if not raw_docs:
            # Queue was empty this round
            idle_elapsed += 1
            if idle_shutdown_after and idle_elapsed >= idle_shutdown_after:
                log.info("Idle for %ds — shutting down.", idle_shutdown_after)
                break
            if idle_elapsed % 10 == 0:
                log.info("Waiting for documents... idle=%ds  queue=%s", idle_elapsed, queue_name)
            time.sleep(1)
            continue

        idle_elapsed = 0

        # ── Parse each raw JSON doc in the batch ──────────────────────────
        for raw in raw_docs:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed JSON: %s — %s", raw[:120], exc)
                continue

            text = payload.get("text", "").strip()
            if not text:
                continue

            yield Document(
                text=text,
                source=payload.get("url", payload.get("source", "redis")),
                doc_id=payload.get(
                    "id",
                    hashlib.sha256(text.encode()).hexdigest()[:16],
                ),
                metadata={
                    "url":   payload.get("url", ""),
                    "title": payload.get("title", ""),
                },
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 56)
    log.info("Dataset Worker starting")
    log.info("  Redis          : %s:%d  db=%d", REDIS_HOST, REDIS_PORT, REDIS_DB)
    log.info("  Input queue    : %s", INPUT_QUEUE)
    log.info("  Output dir     : %s", OUTPUT_DIR)
    log.info("  Redis batch    : %d docs per round-trip", REDIS_BATCH_SIZE)
    log.info("  Shard size     : %d docs per Parquet file", SHARD_SIZE)
    log.info("  Idle shutdown  : %ds", IDLE_SHUTDOWN)
    log.info("=" * 56)

    client = _connect_with_backoff()

    builder = DatasetBuilder(
        output_dir=OUTPUT_DIR,
        min_chars=30,
        min_words=10,
        min_quality_score=0.40,
        min_lang_confidence=0.05,
        similarity_threshold=0.80,
        shard_size=SHARD_SIZE,
        compression="zstd",
        extra_metadata={"pipeline": "slip_talk_collector"},
    )

    doc_stream = _batch_stream(
        client=client,
        queue_name=INPUT_QUEUE,
        batch_size=REDIS_BATCH_SIZE,
        idle_shutdown_after=IDLE_SHUTDOWN,
    )

    stats = builder.run(doc_stream, log_every=1_000)

    log.info("Done.\n%s", stats.report())


if __name__ == "__main__":
    main()