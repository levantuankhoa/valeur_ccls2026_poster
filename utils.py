"""
Valeur — shared utility functions.

Provides: lexicon loading, SBERT initialization, sentence segmentation,
numerical helpers. No pipeline logic lives here.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import config

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_global_seed(seed: int = config.RANDOM_STATE) -> None:
    """Seed numpy, random, and (if available) torch for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# -----------------------------------------------------------------------------
# Device detection & SBERT init
# -----------------------------------------------------------------------------
def detect_device() -> str:
    """Return 'cuda' if a GPU is available and not disabled via FORCE_CPU."""
    if os.getenv("FORCE_CPU", "0").lower() in ("1", "true", "yes"):
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def load_sbert(model_name: str = config.SBERT_MODEL, device: str | None = None):
    """Initialize a SentenceTransformer on the best available device."""
    from sentence_transformers import SentenceTransformer
    device = device or detect_device()
    print(f"[SBERT] loading '{model_name}' on {device}")
    return SentenceTransformer(model_name, device=device)


def encode_texts(
    sbert,
    texts: List[str],
    batch_size: int = config.BATCH_SIZE,
    normalize: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode texts with SBERT; returns (n, dim) numpy array."""
    return sbert.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        batch_size=batch_size,
        show_progress_bar=show_progress,
    )


# -----------------------------------------------------------------------------
# Lexicon loading
# -----------------------------------------------------------------------------
def load_lexicon(lexicon_name: str, path: Path) -> pd.DataFrame:
    """
    Load a VAD lexicon and return a standardized DataFrame with columns
    ['word', 'V', 'A', 'D'].

    Supports the two configured lexicons (warriner, nrc); add new entries
    to config.LEXICONS to extend.
    """
    if lexicon_name not in config.LEXICONS:
        raise ValueError(
            f"Unknown lexicon '{lexicon_name}'. "
            f"Known: {list(config.LEXICONS)}"
        )
    spec = config.LEXICONS[lexicon_name]

    if spec["format"] == "csv":
        df = pd.read_csv(path, encoding="utf-8")
        df.columns = [c.strip() for c in df.columns]
        out = pd.DataFrame({
            "word": df[spec["word_col"]].astype(str).str.lower().str.strip(),
            "V":    pd.to_numeric(df[spec["v_col"]], errors="coerce"),
            "A":    pd.to_numeric(df[spec["a_col"]], errors="coerce"),
            "D":    pd.to_numeric(df[spec["d_col"]], errors="coerce"),
        })
    elif spec["format"] == "tsv":
        # Positional columns (NRC has no header)
        df = pd.read_csv(path, sep="\t", header=None, encoding="utf-8")
        out = pd.DataFrame({
            "word": df.iloc[:, spec["word_col"]].astype(str).str.lower().str.strip(),
            "V":    pd.to_numeric(df.iloc[:, spec["v_col"]], errors="coerce"),
            "A":    pd.to_numeric(df.iloc[:, spec["a_col"]], errors="coerce"),
            "D":    pd.to_numeric(df.iloc[:, spec["d_col"]], errors="coerce"),
        })
    else:
        raise ValueError(f"Unsupported format: {spec['format']}")

    out = out.dropna().reset_index(drop=True)
    out = out[out["word"].str.len() >= config.MIN_WORD_LENGTH].reset_index(drop=True)
    print(f"[lexicon:{lexicon_name}] loaded {len(out):,} items (scale={spec['scale']})")
    return out


# -----------------------------------------------------------------------------
# Sentence segmentation (spaCy)
# -----------------------------------------------------------------------------
def load_spacy():
    """Load en_core_web_sm, auto-downloading if missing."""
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
    except OSError:
        print("[spacy] downloading en_core_web_sm ...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
    nlp.max_length = 2_000_000
    return nlp


def segment_sentences(text_path: Path, min_length: int = config.MIN_SENTENCE_LENGTH) -> List[str]:
    """Read a text file and return a list of sentences (filtered by min length)."""
    if not text_path.exists():
        print(f"[ERROR] text file not found: {text_path}")
        sys.exit(1)
    nlp = load_spacy()
    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read().replace("\n", " ")
    sents = [s.text.strip() for s in nlp(raw).sents if len(s.text.strip()) > min_length]
    print(f"[spacy] segmented {len(sents):,} sentences from {text_path.name}")
    return sents


# -----------------------------------------------------------------------------
# Signal helpers
# -----------------------------------------------------------------------------
def choose_odd_window(length: int, preferred: int = config.SAVGOL_WINDOW_MAX) -> int | None:
    """Return the largest odd Savitzky-Golay window ≤ preferred that fits the signal."""
    if length < 3:
        return None
    wl = min(preferred, length)
    if wl % 2 == 0:
        wl -= 1
    return max(wl, 3)


def make_sliding_windows(
    sentences: List[str],
    window_size: int = config.WINDOW_SIZE,
    step: int = config.STEP,
) -> Tuple[List[str], List[dict]]:
    """Build 3-sentence sliding windows; return (window_texts, metadata)."""
    n_windows = max(0, len(sentences) - window_size + 1)
    texts, meta = [], []
    for i in range(0, n_windows, step):
        chunk = sentences[i: i + window_size]
        texts.append(" ".join(chunk))
        meta.append({
            "Window_ID":       i + 1,
            "Center_Sentence": chunk[window_size // 2],
            "Full_Window_Text": " ".join(chunk),
        })
    return texts, meta
