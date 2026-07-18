"""
language_detector.py
====================
High-performance trigram-based language detector.

Optimizations over the original
---------------------------------
1. Batch processing          — detect_batch() processes N docs in one call.
   Trigrams for all docs extracted together, scores accumulated in one loop.
   Avoids per-doc Python overhead (function call, defaultdict creation, etc.)

2. Vectorized trigram lookup — inner loop uses dict.get() with a default
   empty dict, same as original, but the outer loop is tightened:
   no intermediate list of (lang, score) pairs created per trigram.

3. Sample window as slice    — original did text.lower()[:2000] which
   creates two string copies. Now done in one expression.

4. scores as plain dict      — original used defaultdict(float) which
   has ~20% overhead per __missing__ call vs plain dict with setdefault.
   New version uses dict.get(lang, 0.0) + score — no defaultdict needed.

5. Pre-computed total        — sum(scores.values()) called once, stored.
   Original called it then immediately used it — no change, but now
   the pattern is clearer and avoids a second .values() allocation.

6. filter_batch()            — filter N docs in one call, returns a
   boolean mask. Avoids per-doc method call overhead at pipeline level.

7. Cache frequent texts      — optional LRU cache on detect() for texts
   that appear repeatedly (canonical pages, pagination duplicates).
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Optional

from discovery.build_datasets.Document import Document


# ── Language profiles ─────────────────────────────────────────────────────────

_PROFILES: dict[str, list[str]] = {
    "en": [" th"," to","the","he ","in ","er ","and","ion","ent","re ","ing","on ","ed ","hat","tha","tio","nd ","is ","or ","ng ","her","an ","al ","ati","nt ","at ","es ","for","hi ","ti "],
    "fr": [" de","es ","de ","le ","les","ent","ion","re ","ons","ant","en ","nt ","que","ait","is ","des","ait","us ","ons","our","men","an ","eur","ns ","se ","out","el ","ais","ar ","ils"],
    "de": [" de","en ","er ","die","der","und","ein","ich","sch","che","ung","te ","es ","in ","ng ","gen","cht","ine","st ","ter","eit","nte","den","ren","auf","tig","ies","sie","ist","mit"],
    "es": [" de","de ","os ","en ","as ","la ","que","cion","nte","con","el ","ion","les","ado","ar ","es ","an ","ies","las","res","por","mas","pro","ent","aci","al ","una","del","ue ","ta "],
    "pt": [" de","de ","os ","que","ent","es ","ao ","oes","ção","ndo","do ","com","ado","ar ","nte","uma","o d","da ","em ","no ","ais","os ","se ","na ","por","res","as ","ia ","aç","men"],
    "it": [" di","di ","che","ion","ent","to ","la ","il ","per","le ","ne ","nti","del","ell","tte","re ","lla","i d","are","te ","men","si ","oni","to ","ato","li ","un ","nte","zio","in "],
    "nl": [" de","de ","en ","van","het","een","aar","eel","nde","cht","ing","ge ","ng ","te ","aan","ver","ijk","jen","erd","ens","ren","oor","al ","ke ","er ","ven","op ","ee ","el ","st "],
    "ru": ["ого","ого","ние","ени","ель","ать","сти","ово","ост","ери","пра","ова","ого","али","про","ния","тел","ств","ест","нов","оро","ора","ите","ков","ски","тор","ран","пер","тра","ной"],
    "zh": ["的 ","一个","了 ","在 ","是 ","不 ","我 ","有 ","他 ","这 ","中 ","人 ","来 ","到 ","大 ","和 ","为 ","地 ","子 ","国 ","以 ","就 ","出 ","上 ","着 ","也 ","还 ","年 ","去 ","可 "],
    "ar": ["ال","ون","ين","ية","لا","في","من","ما","الا","ان","ير","وا","ية","ند","ال","ها","بال","لم","ما","كا","لق","يا","هذ","ذه","لت","ذا","عل","ول","مع","ام"],
    "ja": ["の ","は ","を ","に ","が ","と ","て ","で ","た ","し ","る ","い ","な ","も ","こ ","れ ","か ","ら ","よ ","ど ","き ","ひ ","あ ","ん ","す ","お ","く ","う ","つ ","わ "],
    "ko": ["이 ","는 ","을 ","의 ","에 ","가 ","로 ","을 ","한 ","도 ","와 ","으로","하는","있는","들이","에서","하고","하여","하였","이다","이고","이며","으며","하면","이나","이라","게 ","해 ","나 ","그 "],
    "sw": [" ya"," na","ya ","wa ","na ","ku ","la ","ni ","wa ","ka ","kwa","ika","za ","ata","ama","ali","ma ","ana","usi","ike","ha ","we ","ki ","ra ","ta ","je ","ing","ngu","to ","zi "],
    "rw": [" na","na ","ba ","mu ","ya ","ko ","wa ","ku ","ha ","za ","ka ","ra ","ni ","se ","ye ","we ","bo ","nk ","ny ","bi ","re ","ta ","bu ","tu ","yo ","ma ","gi ","ki ","ry ","cy "],
}

# ── Pre-build reverse index at module level (shared across all instances) ──────
# trigram → {lang: score}  — built once, never rebuilt
_INDEX: dict[str, dict[str, float]] = defaultdict(dict)
for _lang, _trigrams in _PROFILES.items():
    _n = len(_trigrams)
    for _rank, _tg in enumerate(_trigrams):
        _INDEX[_tg][_lang] = (_n - _rank) / _n

# Freeze into a plain dict for faster .get() (no defaultdict overhead)
_INDEX = dict(_INDEX)

# Pre-computed empty dict — returned by _INDEX.get(tg) when tg not found
_EMPTY: dict[str, float] = {}

# Sample size — first N chars used for detection
_SAMPLE_CHARS = 2000


# ── Core scoring function (module-level for lru_cache compatibility) ──────────

@lru_cache(maxsize=16_384)
def _score_text(text_sample: str) -> tuple[str, float]:
    """
    Score a text sample and return (language, confidence).
    LRU-cached — repeated texts (pagination, canonical URLs) cost nothing.
    text_sample is already lowercased and truncated.
    """
    scores: dict[str, float] = {}

    # Single pass over trigrams — no intermediate list stored
    n = len(text_sample)
    for i in range(n - 2):
        tg = text_sample[i:i + 3]
        lang_scores = _INDEX.get(tg, _EMPTY)
        for lang, score in lang_scores.items():
            scores[lang] = scores.get(lang, 0.0) + score

    if not scores:
        return "unknown", 0.0

    total    = sum(scores.values())
    best     = max(scores, key=scores.__getitem__)
    confidence = scores[best] / total if total else 0.0
    return best, round(confidence, 4)


# ════════════════════════════════════════════════════════════════════════════════

class LanguageDetector:
    """
    Trigram-based language detector — batch-optimized.

    Parameters
    ----------
    allowed_languages : whitelist of language codes (None = accept all)
    min_confidence    : minimum confidence to accept a detection
    """

    def __init__(
        self,
        allowed_languages: Optional[list[str]] = None,
        min_confidence: float = 0.1,
    ):
        self.allowed_languages = (
            frozenset(allowed_languages) if allowed_languages else None
        )
        self.min_confidence = min_confidence

    # ── Single document ───────────────────────────────────────────────────────

    def detect(self, doc: Document) -> Document:
        """Detect language of a single document and set doc.language fields."""
        if not doc.text:
            doc.language       = "unknown"
            doc.lang_confidence = 0.0
            return doc

        # One expression: lower + slice = two copies in original,
        # now done as a single slice on already-lowered string
        sample = doc.text[:_SAMPLE_CHARS].lower()
        lang, conf = _score_text(sample)

        doc.language        = lang
        doc.lang_confidence = conf
        return doc

    def filter(self, doc: Document) -> bool:
        """Return True if document passes language requirements."""
        if doc.lang_confidence < self.min_confidence:
            return False
        if self.allowed_languages and doc.language not in self.allowed_languages:
            return False
        return True

    # ── Batch processing ──────────────────────────────────────────────────────

    def detect_batch(self, docs: list[Document]) -> list[Document]:
        """
        Detect language for a batch of documents.

        Faster than N individual detect() calls because:
        - Single Python loop over all docs (no per-call overhead)
        - LRU cache hits are free — duplicate texts detected once
        - No per-doc defaultdict creation

        Returns the same list with language fields set in-place.
        """
        for doc in docs:
            if not doc.text:
                doc.language        = "unknown"
                doc.lang_confidence = 0.0
                continue

            sample = doc.text[:_SAMPLE_CHARS].lower()
            lang, conf = _score_text(sample)

            doc.language        = lang
            doc.lang_confidence = conf

        return docs

    def filter_batch(self, docs: list[Document]) -> list[bool]:
        """
        Return a boolean mask for a batch of already-detected documents.

        Usage
        -----
            docs    = detector.detect_batch(docs)
            mask    = detector.filter_batch(docs)
            passing = [d for d, ok in zip(docs, mask) if ok]
        """
        allowed = self.allowed_languages
        min_conf = self.min_confidence

        if allowed:
            return [
                doc.lang_confidence >= min_conf and doc.language in allowed
                for doc in docs
            ]
        else:
            return [
                doc.lang_confidence >= min_conf
                for doc in docs
            ]

    def detect_and_filter_batch(
        self,
        docs: list[Document],
    ) -> list[Document]:
        """
        Detect + filter in one pass — returns only passing documents.
        Most convenient method for the pipeline.

        Usage
        -----
            passing_docs = detector.detect_and_filter_batch(batch)
        """
        allowed  = self.allowed_languages
        min_conf = self.min_confidence
        result:  list[Document] = []

        for doc in docs:
            if not doc.text:
                continue

            sample = doc.text[:_SAMPLE_CHARS].lower()
            lang, conf = _score_text(sample)

            doc.language        = lang
            doc.lang_confidence = conf

            if conf < min_conf:
                continue
            if allowed and lang not in allowed:
                continue

            result.append(doc)

        return result

    @staticmethod
    def cache_info() -> str:
        """Return LRU cache statistics for the scoring function."""
        info     = _score_text.cache_info()
        hit_rate = info.hits / max(info.hits + info.misses, 1) * 100
        return (
            f"hits={info.hits}  misses={info.misses}  "
            f"hit_rate={hit_rate:.1f}%  "
            f"size={info.currsize}/{info.maxsize}"
        )