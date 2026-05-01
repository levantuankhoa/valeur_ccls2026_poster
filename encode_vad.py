#!/usr/bin/env python3
"""
Valeur — Step 2: apply a trained Ridge model to a text corpus.

Reads a plain-text file, segments it into sentences, builds 3-sentence
sliding windows, encodes them with SBERT, and projects each window into
V/A/D via the pre-trained Ridge regressor. Applies Savitzky–Golay
smoothing and writes the result to CSV.

Example:
    python encode_vad.py --text data/metamorphosis.txt \
        --model models/ridge_warriner.joblib \
        --out results/windows_warriner.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import config
from utils import (
    choose_odd_window,
    encode_texts,
    load_sbert,
    make_sliding_windows,
    segment_sentences,
    set_global_seed,
)


def encode(text_path: Path, model_path: Path, out_path: Path) -> pd.DataFrame:
    t0 = time.time()
    set_global_seed()

    # 1. Load trained artefact
    print(f"[load] {model_path}")
    art = joblib.load(model_path)
    models   = art["models"]
    x_scaler = art["x_scaler"]
    y_scaler = art["y_scaler"]
    lex_name = art["lexicon_name"]
    sbert_name = art["sbert_model"]
    print(f"       lexicon={lex_name}  scale={art['scale']}  sbert={sbert_name}")

    # 2. Segment text and build windows
    sentences = segment_sentences(text_path)
    win_texts, win_meta = make_sliding_windows(sentences)
    print(f"[windows] {len(win_texts):,} × {config.WINDOW_SIZE}-sentence windows (stride={config.STEP})")

    # 3. Encode with the same SBERT model used at training time
    sbert = load_sbert(sbert_name)
    emb = encode_texts(sbert, win_texts)

    # 4. Project into V/A/D via Ridge
    emb_s = x_scaler.transform(emb)
    preds_s = np.column_stack([m.predict(emb_s) for m in models])
    vad = y_scaler.inverse_transform(preds_s)

    # 5. Assemble dataframe
    df = pd.DataFrame(win_meta)
    df["Valence"]   = vad[:, 0]
    df["Arousal"]   = vad[:, 1]
    df["Dominance"] = vad[:, 2]
    df["Narrative_Progress"] = np.linspace(0, 1, len(df))

    # 6. Savitzky–Golay smoothing
    wl = choose_odd_window(len(df))
    if wl is not None:
        for col in ("Valence", "Arousal", "Dominance"):
            df[f"{col}_Smooth"] = savgol_filter(
                df[col].values,
                window_length=wl,
                polyorder=config.SAVGOL_POLYORDER,
            )
        print(f"[smooth] Savitzky-Golay wl={wl} polyorder={config.SAVGOL_POLYORDER}")
    else:
        for col in ("Valence", "Arousal", "Dominance"):
            df[f"{col}_Smooth"] = df[col].values
        print("[smooth] signal too short — skipped")

    # 7. Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[save] {out_path}  ({len(df):,} rows)")
    print(f"[done] elapsed {time.time() - t0:.1f}s")
    return df


def main():
    ap = argparse.ArgumentParser(description="Encode text → VAD windows (Valeur step 2/3)")
    ap.add_argument("--text",  required=True, type=Path, help="plain-text corpus file")
    ap.add_argument("--model", required=True, type=Path, help="trained .joblib from train_vad.py")
    ap.add_argument("--out",   required=True, type=Path, help="output windows CSV")
    args = ap.parse_args()
    encode(args.text, args.model, args.out)


if __name__ == "__main__":
    main()
