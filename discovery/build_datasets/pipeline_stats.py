from dataclasses import dataclass

@dataclass
class PipelineStats:
    total_input: int = 0
    after_cleaning: int = 0
    after_language_filter: int = 0
    after_quality_filter: int = 0
    after_dedup: int = 0
    written: int = 0
    bytes_written: int = 0
    elapsed_seconds: float = 0.0

    def report(self) -> str:
        lines = [
            "┌─────────────────────────────────────────┐",
            "│          PIPELINE STATISTICS            │",
            "├─────────────────────────────────────────┤",
            f"│  Input documents      : {self.total_input:>10,}      │",
            f"│  After cleaning       : {self.after_cleaning:>10,}      │",
            f"│  After lang filter    : {self.after_language_filter:>10,}      │",
            f"│  After quality filter : {self.after_quality_filter:>10,}      │",
            f"│  After deduplication  : {self.after_dedup:>10,}      │",
            f"│  Written to disk      : {self.written:>10,}      │",
            f"│  Bytes written        : {self.bytes_written:>10,}      │",
            f"│  Elapsed (s)          : {self.elapsed_seconds:>10.2f}      │",
            "└─────────────────────────────────────────┘",
        ]
        return "\n".join(lines)