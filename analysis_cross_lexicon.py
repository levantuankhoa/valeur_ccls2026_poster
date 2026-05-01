#!/usr/bin/env python3
"""
Valeur — supplementary analysis: trajectory-level V/A/D agreement.

Computes Spearman rho and Pearson r between the smoothed V, A, D
trajectories produced by encode_vad.py for two lexicons (typically
Warriner and NRC), separately for each affective dimension. This is
the trajectory-level analogue of Mohammad (2025)'s word-level
NRC-vs-Warriner overlap correlations and is the headline cross-
validation argument for the dual-lexicon design.

Example:
    python analysis_cross_lexicon.py \
        --a results/windows_warriner.csv --label-a warriner \
        --b results/windows_nrc.csv      --label-b nrc \
        --out results/cross_lexicon_agreement.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

DIMENSIONS = ("Valence", "Arousal", "Dominance")


def compute_agreement(
    path_a: Path,
    path_b: Path,
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    df_a = pd.read_csv(path_a, encoding="utf-8-sig")
    df_b = pd.read_csv(path_b, encoding="utf-8-sig")

    # Align on minimum length (sentence segmentation can yield ±1 window
    # difference between lexicons because of slight scale-dependent
    # filtering, even on the same corpus).
    L = min(len(df_a), len(df_b))
    df_a = df_a.iloc[:L].reset_index(drop=True)
    df_b = df_b.iloc[:L].reset_index(drop=True)

    print(f"\n{'=' * 64}")
    print(f"Cross-lexicon trajectory agreement: {label_a} vs {label_b}")
    print(f"  Aligned length: {L:,} windows")
    print(f"{'=' * 64}")
    print(f"  {'Dimension':<12} {'Spearman rho':>14} {'p (Spearman)':>14} "
          f"{'Pearson r':>12} {'p (Pearson)':>13}")
    print("  " + "-" * 70)

    rows = []
    for dim in DIMENSIONS:
        col = f"{dim}_Smooth"
        if col not in df_a.columns or col not in df_b.columns:
            raise KeyError(
                f"Column '{col}' missing from one of the inputs. "
                f"Did you pass the encode_vad.py output (windows CSV)?"
            )
        x = df_a[col].astype(float).values
        y = df_b[col].astype(float).values

        rho, p_rho = spearmanr(x, y)
        r, p_r = pearsonr(x, y)
        print(f"  {dim:<12} {rho:>+14.3f} {p_rho:>14.2e} "
              f"{r:>+12.3f} {p_r:>13.2e}")
        rows.append({
            "dimension":     dim,
            "lexicon_a":     label_a,
            "lexicon_b":     label_b,
            "n_windows":     L,
            "spearman_rho":  float(rho),
            "spearman_p":    float(p_rho),
            "pearson_r":     float(r),
            "pearson_p":     float(p_r),
        })
    print()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Cross-lexicon trajectory agreement on V/A/D smoothed signals.",
    )
    ap.add_argument("--a", required=True, type=Path,
                    help="windows CSV for lexicon A (from encode_vad.py)")
    ap.add_argument("--b", required=True, type=Path,
                    help="windows CSV for lexicon B (from encode_vad.py)")
    ap.add_argument("--label-a", default="A", help="lexicon A label for reporting")
    ap.add_argument("--label-b", default="B", help="lexicon B label for reporting")
    ap.add_argument("--out", type=Path, default=None,
                    help="optional CSV path to persist the agreement table")
    args = ap.parse_args()

    df = compute_agreement(args.a, args.b, args.label_a, args.label_b)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
