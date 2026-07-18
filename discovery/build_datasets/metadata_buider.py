from __future__ import annotations

import re
from typing import Optional
from discovery.build_datasets.Document import Document
import hashlib
from datetime import datetime, timezone


class MetadataBuilder:


    PIPELINE_VERSION = "1.0.0"

    _SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, extra_fields: Optional[dict] = None):
        self.extra_fields = extra_fields or {}

    def build(self, doc: Document) -> Document:
        text = doc.text
        sentences = [s for s in self._SENT_SPLIT.split(text) if s.strip()]
        n_sent = max(len(sentences), 1)
        avg_sent_len = round(doc.word_count / n_sent, 2)

        chars_alpha_num = sum(1 for c in text if c.isalnum())
        punct_ratio = round(
            sum(1 for c in text if c in ".,!?;:\"'()[]{}") / max(len(text), 1), 4
        )
        numeric_ratio = round(
            sum(1 for c in text if c.isdigit()) / max(len(text), 1), 4
        )
        content_hash = hashlib.sha256(text.encode()).hexdigest()

        doc.metadata.update({
            "pipeline_version": self.PIPELINE_VERSION,
            "build_timestamp": datetime.now(timezone.utc).isoformat(),
            "source": doc.source,
            "url": doc.metadata.get("url", doc.source),
            "title": doc.metadata.get("title", ""),
            "doc_id": doc.doc_id,
            "language": doc.language,
            "lang_confidence": doc.lang_confidence,
            "quality_score": doc.quality_score,
            "char_count": doc.char_count,
            "word_count": doc.word_count,
            "token_estimate": doc.token_estimate,
            "sentence_count": n_sent,
            "avg_sentence_len": avg_sent_len,
            "punctuation_ratio": punct_ratio,
            "numeric_ratio": numeric_ratio,
            "content_hash": content_hash,
            **self.extra_fields,
        })
        return doc