"""Hybrid retrieval over MCU datasheets: BM25 (exact part numbers, units) + dense embeddings.

Datasheets mix prose with wide tables (ordering information, memory maps). Pages are
extracted with ``pdftotext -layout`` so table rows stay on one line, column gaps are
turned into `` | `` separators, and pages are cut into overlapping line-based chunks.

Retrieval fuses a BM25 ranking, which matches exact tokens such as "LPC55S69JBD100"
or "Kbytes", with a dense ranking (BAAI/bge-small-en-v1.5, runs locally on CPU) that
matches paraphrases ("on-chip RAM" for "SRAM"), using reciprocal rank fusion.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import yaml
from rank_bm25 import BM25Okapi

from edge_deploy_agent.tools.deploy import ROOT

DATASHEETS_DIR = ROOT / "data" / "datasheets"
INDEX_DIR = ROOT / "artifacts" / "rag_index"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_chips() -> dict[str, dict]:
    return yaml.safe_load((ROOT / "configs" / "chips.yaml").read_text())


def pdf_pages(path: Path) -> list[str]:
    """Text of each page; table columns separated by ' | '."""
    text = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"], check=True, capture_output=True, text=True
    ).stdout
    pages = text.split("\f")
    return [re.sub(r" {3,}", " | ", p) for p in pages]


def chunk_page(text: str, max_chars: int = 1000, overlap_lines: int = 3) -> list[str]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    chunks, current = [], []
    for line in lines:
        current.append(line)
        if sum(len(x) + 1 for x in current) >= max_chars:
            chunks.append("\n".join(current))
            current = current[-overlap_lines:]
    if current and (not chunks or len(current) > overlap_lines):
        chunks.append("\n".join(current))
    return chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def build_index(index_dir: Path = INDEX_DIR) -> int:
    """Extract, chunk and embed every datasheet listed in configs/chips.yaml."""
    chips = load_chips()
    doc_chips: dict[str, list[str]] = {}
    for chip, cfg in chips.items():
        for doc in cfg["datasheets"]:
            doc_chips.setdefault(doc, []).append(chip)

    chunks = []
    for doc, chip_ids in doc_chips.items():
        path = DATASHEETS_DIR / doc
        if not path.exists():
            raise FileNotFoundError(f"Missing datasheet {path} (see README: datasheets are not versioned)")
        for page_no, page in enumerate(pdf_pages(path), start=1):
            for text in chunk_page(page):
                chunks.append({"doc": doc, "page": page_no, "chips": chip_ids, "text": text})

    embeddings = _embedder().encode(
        [c["text"] for c in chunks], batch_size=64, normalize_embeddings=True, show_progress_bar=True
    )
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "chunks.json").write_text(json.dumps(chunks))
    np.save(index_dir / "embeddings.npy", embeddings.astype(np.float32))
    return len(chunks)


@dataclass
class Hit:
    doc: str
    page: int
    text: str
    score: float


class DatasheetIndex:
    def __init__(self, index_dir: Path = INDEX_DIR) -> None:
        if not (index_dir / "chunks.json").exists():
            raise FileNotFoundError("No index found. Run: python scripts/build_index.py")
        self.chunks = json.loads((index_dir / "chunks.json").read_text())
        self.embeddings = np.load(index_dir / "embeddings.npy")

    @cached_property
    def _model(self):
        return _embedder()

    def search(self, query: str, chip: str | None = None, k: int = 5, pool: int = 50, rrf_k: int = 60) -> list[Hit]:
        """Top-``k`` chunks for ``query``, restricted to ``chip``'s datasheets if given."""
        ids = [i for i, c in enumerate(self.chunks) if chip is None or chip in c["chips"]]
        if not ids:
            return []

        bm25 = BM25Okapi([_tokenize(self.chunks[i]["text"]) for i in ids])
        bm25_rank = np.argsort(-bm25.get_scores(_tokenize(query)))[:pool]

        q = self._model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        dense_rank = np.argsort(-(self.embeddings[ids] @ q))[:pool]

        fused: dict[int, float] = {}
        for ranking in (bm25_rank, dense_rank):
            for rank, local in enumerate(ranking):
                fused[local] = fused.get(local, 0.0) + 1.0 / (rrf_k + rank + 1)

        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [
            Hit(self.chunks[ids[i]]["doc"], self.chunks[ids[i]]["page"], self.chunks[ids[i]]["text"], fused[i])
            for i in best
        ]
