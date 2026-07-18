from __future__ import annotations

from discovery.build_datasets.Document import Document
from discovery.build_datasets.text_cleaner import TextCleaner
from discovery.build_datasets.language_detector import LanguageDetector
from discovery.build_datasets.quality_filter import QualityFilter
from discovery.build_datasets.deduplicator import Deduplicator
from discovery.build_datasets.metadata_buider import MetadataBuilder
from discovery.build_datasets.parquet_writer import ParquetWriter
from discovery.build_datasets.pipeline_stats import PipelineStats
from pathlib import Path
from typing import Optional, Callable, Iterable, Iterator
import time
import json
import hashlib
import logging

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("dataset_builder")

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_PYARROW = True
except ImportError:
    _HAVE_PYARROW = False

class DatasetBuilder:
    """
    Full tokenization-dataset pipeline.

    Usage
    -----
    >>> builder = DatasetBuilder(output_dir="my_dataset")
    >>> builder.run(my_document_iterator)
    """

    def __init__(
        self,
        output_dir: str | Path = "dataset_output",
        *,
        # Cleaner
        min_chars: int = 50,
        max_chars: int = 500_000,
        strip_urls: bool = True,
        strip_html: bool = True,
        # Language
        allowed_languages: Optional[list[str]] = None,
        min_lang_confidence: float = 0.10,
        # Quality
        min_quality_score: float = 0.50,
        min_words: int = 20,
        # Dedup
        similarity_threshold: float = 0.80,
        num_hashes: int = 128,
        # Writer
        shard_size: int = 10_000,
        compression: str = "zstd",
        # Metadata
        extra_metadata: Optional[dict] = None,
        # Hooks
        pre_filter_hook: Optional[Callable[[Document], Optional[Document]]] = None,
    ):
        self.cleaner = TextCleaner(
            min_chars=min_chars,
            max_chars=max_chars,
            strip_urls=strip_urls,
            strip_html=strip_html,
        )
        self.detector = LanguageDetector(
            allowed_languages=allowed_languages,
            min_confidence=min_lang_confidence,
        )
        self.quality_filter = QualityFilter(
            min_score=min_quality_score,
            min_words=min_words,
        )
        self.deduplicator = Deduplicator(
            similarity_threshold=similarity_threshold,
            num_hashes=num_hashes,
        )
        self.meta_builder = MetadataBuilder(extra_fields=extra_metadata or {})
        self.writer = ParquetWriter(
            output_dir=output_dir,
            shard_size=shard_size,
            compression=compression,
        )
        self.pre_filter_hook = pre_filter_hook
        self.stats = PipelineStats()

    # ── Pipeline stages ─────────────────────────────────────────────

    def _process_one(self, doc: Document) -> Optional[Document]:
        # Optional user hook
        if self.pre_filter_hook:
            doc = self.pre_filter_hook(doc)
            if doc is None:
                return None

        # Stage 1: Clean
        doc = self.cleaner.clean(doc)
        if doc is None:
            return None
        self.stats.after_cleaning += 1

        # Stage 2: Language detect + filter
        doc = self.detector.detect(doc)
        if not self.detector.filter(doc):
            return None
        self.stats.after_language_filter += 1

        # Stage 3: Quality score + filter
        doc = self.quality_filter.score(doc)
        if not self.quality_filter.filter(doc):
            return None
        self.stats.after_quality_filter += 1

        # Stage 4: Deduplication
        if self.deduplicator.is_duplicate(doc):
            return None
        self.stats.after_dedup += 1

        # Stage 5: Metadata
        doc = self.meta_builder.build(doc)

        return doc

    # ── Main entry point ─────────────────────────────────────────────

    def run(
        self,
        documents: Iterable[Document],
        *,
        log_every: int = 1000,
    ) -> PipelineStats:
        t0 = time.perf_counter()
        log.info("Dataset builder started.")
        log.info("pyarrow available: %s", _HAVE_PYARROW)

        with self.writer:
            for i, doc in enumerate(documents, 1):
                self.stats.total_input += 1

                out = self._process_one(doc)
                if out is not None:
                    self.writer.write(out)

                if i % log_every == 0:
                    log.info(
                        "Processed %d  |  passed dedup %d  |  written %d",
                        i, self.stats.after_dedup, self.writer._total_written,
                    )

            written, nbytes = self.writer.close()

        self.stats.written = written
        self.stats.bytes_written = nbytes
        self.stats.elapsed_seconds = round(time.perf_counter() - t0, 2)

        log.info("Pipeline finished.\n%s", self.stats.report())
        return self.stats

    def run_from_redis(
        self,
        queue_name: str = "clean_text_queue",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        *,
        block_timeout_seconds: int = 5,
        idle_shutdown_after: int = 30,
        log_every: int = 1000,
    ) -> PipelineStats:
        """
        Consume JSON messages from a Redis list (BLPOP) and run the full pipeline.

        Each message must be a JSON string with at least a "text" field.
        Optional fields recognised: "url", "title", "id".

        The worker shuts down gracefully after `idle_shutdown_after` seconds
        of an empty queue (no new messages).

        Parameters
        ----------
        queue_name          : Redis list key pushed to by worker_extractor
        redis_host/port/db  : Redis connection params
        redis_password      : Redis AUTH password (None = no auth)
        block_timeout_seconds : BLPOP timeout per round-trip
        idle_shutdown_after   : seconds of silence before graceful exit
        log_every             : log progress every N documents consumed
        """
        try:
            import redis as redis_lib
        except ImportError:
            raise ImportError(
                "redis-py is required: pip install redis"
            )

        import signal

        def _make_client():
            c = redis_lib.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
                retry_on_timeout=True,
            )
            c.ping()
            return c

        client = _make_client()
        log.info("Connected to Redis %s:%d  queue='%s'", redis_host, redis_port, queue_name)

        _shutdown = False
        def _handle_sigterm(sig, frame):
            nonlocal _shutdown
            log.info("SIGTERM received — flushing and shutting down.")
            _shutdown = True
        signal.signal(signal.SIGTERM, _handle_sigterm)
        signal.signal(signal.SIGINT,  _handle_sigterm)

        def _redis_stream() -> Iterator[Document]:
            nonlocal client, _shutdown
            idle_elapsed = 0.0
            while not _shutdown:
                try:
                    result = client.blpop(queue_name, timeout=block_timeout_seconds)
                except Exception as exc:
                    log.warning("Redis connection lost (%s) — reconnecting in 3s…", exc)
                    time.sleep(3)
                    try:
                        client = _make_client()
                        log.info("Reconnected to Redis.")
                    except Exception as reconn_exc:
                        log.error("Reconnect failed: %s", reconn_exc)
                    continue
                if result is None:
                    idle_elapsed += block_timeout_seconds
                    log.debug("Queue empty — idle %.0fs / %ds", idle_elapsed, idle_shutdown_after)
                    if idle_elapsed >= idle_shutdown_after:
                        log.info("Idle timeout reached. Shutting down Redis consumer.")
                        break
                    continue

                idle_elapsed = 0.0
                _, raw = result
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    log.warning("Skipping malformed JSON message: %s — %s", raw[:120], exc)
                    continue

                text = payload.get("text", "").strip()
                if not text:
                    log.debug("Skipping message with empty text field.")
                    continue

                yield Document(
                    text=text,
                    source=payload.get("url", payload.get("source", "redis")),
                    doc_id=payload.get("id", hashlib.sha256(text.encode()).hexdigest()[:16]),
                    metadata={
                        "url":   payload.get("url", ""),
                        "title": payload.get("title", ""),
                    },
                )

        return self.run(_redis_stream(), log_every=log_every)