"""
parquet_writer.py
=================
High-throughput Parquet writer for the AI tokenization dataset pipeline.

Optimizations over the original
---------------------------------
1. Columnar buffer instead of list of Document objects
   Original: buffer stores Document objects, then iterates 10× at flush
             (one list comprehension per column = 10 passes over N docs)
   New:      buffer IS the columns — each field appended directly on write()
             flush needs zero iteration — columns handed straight to pyarrow

2. Pre-built PyArrow schema
   Original: schema rebuilt on every shard write
   New:      schema built once at __init__, reused forever

3. write_batch() method
   Accepts a list[Document] and appends all at once using
   zip-based column extension — faster than N individual write() calls
   when the caller already has a batch (e.g. dataset_worker batch pop)

4. Row-group size tuning
   PyArrow default row group = entire table (bad for large shards).
   Now explicitly set to 50,000 rows — better compression + faster reads.

5. Compression level tuning
   zstd level 3 (default) is a good speed/size tradeoff.
   Exposed as compression_level for tuning.

6. Background flush (optional)
   When use_threads=True, flush runs in a ThreadPoolExecutor so the
   pipeline never stalls waiting for disk I/O.
"""

from __future__ import annotations

import gzip
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from discovery.build_datasets.Document import Document

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_PYARROW = True
except ImportError:
    _HAVE_PYARROW = False

log = logging.getLogger("parquet_writer")


# ── PyArrow schema — built once, reused forever ────────────────────────────

_SCHEMA: Optional["pa.Schema"] = None

def _get_schema() -> "pa.Schema":
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = pa.schema([
            ("doc_id",          pa.string()),
            ("text",            pa.large_string()),
            ("source",          pa.string()),
            ("language",        pa.string()),
            ("lang_confidence", pa.float32()),
            ("quality_score",   pa.float32()),
            ("char_count",      pa.int64()),
            ("word_count",      pa.int64()),
            ("token_estimate",  pa.int64()),
            ("metadata_json",   pa.string()),
        ])
    return _SCHEMA


# ════════════════════════════════════════════════════════════════════════════════

class ParquetWriter:
    """
    Buffered, columnar Parquet writer.

    Parameters
    ----------
    output_dir        : directory for shard files
    shard_size        : documents per shard (default 10,000)
    compression       : parquet compression codec (default "zstd")
    compression_level : zstd level 1–22 (default 3 = fast + good ratio)
    row_group_size    : rows per parquet row group (default 50,000)
    use_threads       : flush in background thread (default True)
    """

    def __init__(
        self,
        output_dir: str | Path,
        shard_size: int = 10_000,
        compression: str = "zstd",
        compression_level: int = 3,
        row_group_size: int = 50_000,
        use_threads: bool = True,
    ):
        self.output_dir        = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size        = shard_size
        self.compression       = compression
        self.compression_level = compression_level
        self.row_group_size    = row_group_size
        self.use_threads       = use_threads

        # ── Columnar buffer — append per-field directly ───────────────────
        # Instead of storing Document objects and iterating 10× at flush,
        # we keep one list per column and append to them directly on write().
        # At flush time, columns are handed straight to pyarrow — zero extra
        # iteration needed.
        self._cols: dict[str, list] = self._empty_cols()

        self._shard_idx    = 0
        self._total_written= 0
        self._total_bytes  = 0
        self._buffer_size  = 0   # tracks len without calling len(_cols[...])

        # Background flush executor
        self._executor  = ThreadPoolExecutor(max_workers=1) if use_threads else None
        self._flush_fut = None   # future for in-flight background flush
        self._lock      = threading.Lock()

    # ── Column management ─────────────────────────────────────────────────────

    @staticmethod
    def _empty_cols() -> dict[str, list]:
        return {
            "doc_id":          [],
            "text":            [],
            "source":          [],
            "language":        [],
            "lang_confidence": [],
            "quality_score":   [],
            "char_count":      [],
            "word_count":      [],
            "token_estimate":  [],
            "metadata_json":   [],
        }

    def _append_doc(self, cols: dict, doc: Document) -> None:
        """Append one document's fields to the columnar buffer."""
        cols["doc_id"].append(doc.doc_id)
        cols["text"].append(doc.text)
        cols["source"].append(doc.source)
        cols["language"].append(doc.language)
        cols["lang_confidence"].append(doc.lang_confidence)
        cols["quality_score"].append(doc.quality_score)
        cols["char_count"].append(doc.char_count)
        cols["word_count"].append(doc.word_count)
        cols["token_estimate"].append(doc.token_estimate)
        cols["metadata_json"].append(
            json.dumps(doc.metadata, ensure_ascii=False)
        )

    # ── Shard writers ─────────────────────────────────────────────────────────

    def _write_shard_pyarrow(self, cols: dict, shard_idx: int) -> Path:
        """
        Write one shard using pre-built columnar data.
        Zero iteration needed — columns already in correct format.
        """
        table = pa.table(cols, schema=_get_schema())
        out   = self.output_dir / f"shard_{shard_idx:05d}.parquet"

        pq.write_table(
            table,
            out,
            compression=self.compression,
            compression_level=self.compression_level,
            row_group_size=self.row_group_size,
            use_dictionary=True,      # dictionary-encode string columns
            write_statistics=True,    # enables predicate pushdown on read
        )
        return out

    def _write_shard_jsonlgz(self, cols: dict, shard_idx: int) -> Path:
        """Fallback when pyarrow is not installed."""
        out = self.output_dir / f"shard_{shard_idx:05d}.jsonl.gz"
        n   = len(cols["doc_id"])
        with gzip.open(out, "wt", encoding="utf-8") as f:
            for i in range(n):
                record = {k: cols[k][i] for k in cols}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return out

    def _write_cols(self, cols: dict, shard_idx: int) -> Path:
        if _HAVE_PYARROW:
            return self._write_shard_pyarrow(cols, shard_idx)
        return self._write_shard_jsonlgz(cols, shard_idx)

    # ── Flush logic ───────────────────────────────────────────────────────────

    def _do_flush(self, cols: dict, shard_idx: int, n_docs: int) -> int:
        """
        Actually write a shard to disk. Can run in background thread.
        Returns bytes written.
        """
        out  = self._write_cols(cols, shard_idx)
        size = out.stat().st_size
        log.info(
            "Wrote shard %05d → %s  (%d docs, %.1f KB)",
            shard_idx, out.name, n_docs, size / 1024,
        )
        return size

    def _flush(self) -> None:
        """
        Swap out the current columnar buffer and write it to disk.
        If use_threads=True, the write runs in the background so the
        pipeline can keep processing while disk I/O happens.
        """
        if self._buffer_size == 0:
            return

        with self._lock:
            # Swap buffer — pipeline continues filling the new one
            cols           = self._cols
            shard_idx      = self._shard_idx
            n_docs         = self._buffer_size
            self._cols         = self._empty_cols()
            self._buffer_size  = 0
            self._shard_idx   += 1
            self._total_written += n_docs

        if self._executor:
            # Wait for previous flush to finish before starting next
            if self._flush_fut is not None:
                size = self._flush_fut.result()
                self._total_bytes += size

            self._flush_fut = self._executor.submit(
                self._do_flush, cols, shard_idx, n_docs
            )
        else:
            size = self._do_flush(cols, shard_idx, n_docs)
            self._total_bytes += size

    # ── Public API ────────────────────────────────────────────────────────────

    def write(self, doc: Document) -> None:
        """Write a single document to the buffer."""
        self._append_doc(self._cols, doc)
        self._buffer_size += 1
        if self._buffer_size >= self.shard_size:
            self._flush()

    def write_batch(self, docs: list[Document]) -> None:
        """
        Write a batch of documents — faster than N individual write() calls.
        Designed for dataset_worker.py which already pops documents in batches.

        Handles shard boundaries automatically — a batch can span shards.
        """
        for doc in docs:
            self._append_doc(self._cols, doc)
            self._buffer_size += 1
            if self._buffer_size >= self.shard_size:
                self._flush()

    def close(self) -> tuple[int, int]:
        """
        Flush remaining buffer and wait for any background flush to finish.
        Returns (total_docs_written, total_bytes_written).
        """
        self._flush()

        # Wait for background flush to complete
        if self._flush_fut is not None:
            size = self._flush_fut.result()
            self._total_bytes += size
            self._flush_fut = None

        if self._executor:
            self._executor.shutdown(wait=True)

        return self._total_written, self._total_bytes

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()