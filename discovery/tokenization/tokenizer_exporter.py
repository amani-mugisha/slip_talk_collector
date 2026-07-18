from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("tokenizer_exporter")



class TokenizerExporter:
    """
    Exports a trained tokenizer to all formats needed for AI training.

    Parameters
    ----------
    tokenizer_dir : directory containing the trained tokenizer files
    export_dir    : where to write export packages (default: tokenizer_dir/exports)
    """

    def __init__(
        self,
        tokenizer_dir: str | Path = "tokenizer_output",
        export_dir:    Optional[str | Path] = None,
    ):
        self.tokenizer_dir = Path(tokenizer_dir)
        self.export_dir    = Path(export_dir) if export_dir else \
                             self.tokenizer_dir / "exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)

        self._load_tokenizer()

    # ── Load ──────────────────────────────────────────────────────────────────

    def _load_tokenizer(self):
        from transformers import PreTrainedTokenizerFast

        tok_json = self.tokenizer_dir / "tokenizer.json"
        if not tok_json.exists():
            raise FileNotFoundError(
                f"tokenizer.json not found in '{self.tokenizer_dir}'. "
                "Run tokenizer_trainer.py first."
            )

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(self.tokenizer_dir)
        )
        # Load raw tokenizer.json for vocab/merges extraction
        with open(tok_json, encoding="utf-8") as f:
            self._raw = json.load(f)

        log.info(
            "Loaded tokenizer from '%s'  vocab_size=%d",
            self.tokenizer_dir, self.tokenizer.vocab_size,
        )

    # ── Export 1: HuggingFace (primary) ──────────────────────────────────────

    def export_huggingface(self) -> Path:
        """
        Save a clean HuggingFace PreTrainedTokenizerFast package.
        This is the main format used for model training.
        """
        hf_dir = self.export_dir / "huggingface"
        hf_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer.save_pretrained(str(hf_dir))
        log.info("HuggingFace export → %s/", hf_dir)
        for p in sorted(hf_dir.iterdir()):
            log.info("  %s  (%.1f KB)", p.name, p.stat().st_size / 1024)
        return hf_dir

    # ── Export 2: vocab.txt (one token per line, BERT-style) ─────────────────

    def export_vocab_txt(self) -> Path:
        """
        Write vocab.txt — one token per line sorted by ID.
        Compatible with many training frameworks that expect this format.
        """
        vocab = self.tokenizer.get_vocab()                   # token → id
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])  # sort by id

        out = self.export_dir / "vocab.txt"
        with open(out, "w", encoding="utf-8") as f:
            for token, _ in sorted_vocab:
                f.write(token + "\n")

        log.info("vocab.txt → %s  (%d tokens, %.1f KB)", out, len(sorted_vocab), out.stat().st_size / 1024)
        return out

    # ── Export 3: vocab.json (token → id mapping) ────────────────────────────

    def export_vocab_json(self) -> Path:
        """Write vocab.json — full token→id mapping."""
        vocab = self.tokenizer.get_vocab()
        out   = self.export_dir / "vocab.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        log.info("vocab.json → %s  (%d tokens, %.1f KB)", out, len(vocab), out.stat().st_size / 1024)
        return out

    # ── Export 4: merges.txt (BPE merge rules) ────────────────────────────────

    def export_merges(self) -> Optional[Path]:
        """
        Write merges.txt for BPE tokenizers.
        Returns None if the tokenizer is not BPE.
        """
        model = self._raw.get("model", {})
        merges = model.get("merges", [])

        if not merges:
            log.info("No merges found — tokenizer is not BPE, skipping merges.txt.")
            return None

        out = self.export_dir / "merges.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for merge in merges:
                if isinstance(merge, list):
                    f.write(" ".join(merge) + "\n")
                else:
                    f.write(str(merge) + "\n")

        log.info("merges.txt → %s  (%d rules, %.1f KB)", out, len(merges), out.stat().st_size / 1024)
        return out

    # ── Export 5: tokenizer_config.json (standalone config) ──────────────────

    def export_config(self) -> Path:
        """Write a standalone tokenizer config for documentation."""
        # Load training config if available
        training_config = {}
        tc_path = self.tokenizer_dir / "training_config.json"
        if tc_path.exists():
            with open(tc_path) as f:
                training_config = json.load(f)

        config = {
            "tokenizer_class":  "PreTrainedTokenizerFast",
            "vocab_size":       self.tokenizer.vocab_size,
            "model_max_length": self.tokenizer.model_max_length,
            "bos_token":        self.tokenizer.bos_token,
            "eos_token":        self.tokenizer.eos_token,
            "unk_token":        self.tokenizer.unk_token,
            "pad_token":        self.tokenizer.pad_token,
            "mask_token":       self.tokenizer.mask_token,
            "padding_side":     self.tokenizer.padding_side,
            "truncation_side":  self.tokenizer.truncation_side,
            "training":         training_config,
            "exported_at":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        out = self.export_dir / "tokenizer_config_export.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        log.info("tokenizer_config_export.json → %s", out)
        return out

    # ── Manifest with SHA-256 checksums ──────────────────────────────────────

    def _sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def write_manifest(self, exported_files: list[Path]) -> Path:
        """Write a manifest.json with file sizes and SHA-256 checksums."""
        manifest = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "vocab_size":   self.tokenizer.vocab_size,
            "files": [],
        }
        for p in exported_files:
            if p and p.exists():
                manifest["files"].append({
                    "name":       p.name,
                    "path":       str(p.relative_to(self.export_dir)),
                    "size_bytes": p.stat().st_size,
                    "sha256":     self._sha256(p),
                })

        out = self.export_dir / "manifest.json"
        with open(out, "w") as f:
            json.dump(manifest, f, indent=2)
        log.info("manifest.json → %s", out)
        return out

    # ── Full export run ───────────────────────────────────────────────────────

    def run(self) -> Path:
        """Run all exports and write manifest. Returns export directory."""
        log.info("=" * 56)
        log.info("Tokenizer Exporter")
        log.info("  Source : %s", self.tokenizer_dir)
        log.info("  Target : %s", self.export_dir)
        log.info("=" * 56)

        exported: list[Path] = []

        # Run each export
        hf_dir  = self.export_huggingface()
        exported += list(hf_dir.iterdir())

        exported.append(self.export_vocab_txt())
        exported.append(self.export_vocab_json())

        merges = self.export_merges()
        if merges:
            exported.append(merges)

        exported.append(self.export_config())

        # Manifest
        self.write_manifest(exported)

        # Summary
        total_size = sum(
            p.stat().st_size for p in self.export_dir.rglob("*") if p.is_file()
        )
        log.info("=" * 56)
        log.info("Export complete → %s/", self.export_dir)
        log.info("Total size: %.1f KB", total_size / 1024)
        log.info("=" * 56)

        return self.export_dir


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    exporter = TokenizerExporter(
        tokenizer_dir = os.getenv("TOKENIZER_DIR", "tokenizer_output"),
        export_dir    = os.getenv("EXPORT_DIR",    "tokenizer_output/exports"),
    )
    exporter.run()


if __name__ == "__main__":
    main()
