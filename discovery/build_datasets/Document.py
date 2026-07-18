from dataclasses import dataclass
from dataclasses import dataclass, field

@dataclass
class Document:
    """Single unit flowing through the pipeline."""
    text: str
    source: str = ""
    doc_id: str = ""
    metadata: dict = field(default_factory=dict)

    # Fields written by pipeline stages
    language: str = ""
    lang_confidence: float = 0.0
    quality_score: float = 0.0
    is_duplicate: bool = False
    minhash: list[int] = field(default_factory=list)
    word_count: int = 0
    char_count: int = 0
    token_estimate: int = 0