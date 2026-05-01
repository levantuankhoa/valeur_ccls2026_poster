#!/usr/bin/env python3
"""
Valeur — supplementary analysis: extract consensus entrapment sentences.

For each pair of windows CSVs, computes NEI per lexicon, applies the
same row-normalized top-percentile flagging as run_consensus() in
nei_plot.py, and exports the windows where BOTH lexicons agree
entrapment is present, along with the center sentence for each.

This is the strongest evidence list for the poster — moments in the
text where two independently-constructed lexicons converge on the
same entrapment signature.

Example:
    python analysis_consensus_sentences.py \
        --a results/windows_warriner.csv --label-a warriner \
        --b results/windows_nrc.csv      --label-b nrc \
        --out results/consensus_sentences.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
from nei_plot import compute_nei


def extract_consensus(
    path_a: Path,
    path_b: Path,
    label_a: str,
    label_b: str,
    method: str = config.NEI_METHOD,
    min_component: float = config.NEI_MIN_COMPONENT,
    percentile: float = config.NEI_PERCENTILE,
) -> pd.DataFrame:
    df_a = pd.read_csv(path_a, encoding="utf-8-sig")
    df_b = pd.read_csv(path_b, encoding="utf-8-sig")

    Va = df_a["Valence_Smooth"].astype(float).values
    Aa = df_a["Arousal_Smooth"].astype(float).values
    Da = df_a["Dominance_Smooth"].astype(float).values

    Vb = df_b["Valence_Smooth"].astype(float).values
    Ab = df_b["Arousal_Smooth"].astype(float).values
    Db = df_b["Dominance_Smooth"].astype(float).values

    nei_a, *_ = compute_nei(Va, Aa, Da, method=method, min_component=min_component)
    nei_b, *_ = compute_nei(Vb, Ab, Db, method=method, min_component=min_component)

    # Align on minimum length (mirrors run_consensus alignment logic)
    L = min(len(nei_a), len(nei_b))
    nei_a = nei_a[:L]
    nei_b = nei_b[:L]
    df_a = df_a.iloc[:L].reset_index(drop=True)
    df_b = df_b.iloc[:L].reset_index(drop=True)

    # Row-normalize (same min-max as run_consensus → reproduces the heatmap)
    eps = 1e-9
    norm_a = (nei_a - nei_a.min()) / (np.ptp(nei_a) + eps)
    norm_b = (nei_b - nei_b.min()) / (np.ptp(nei_b) + eps)

    # Top-percentile flagging per lexicon, then intersect
    thresh_a = float(np.quantile(norm_a, percentile))
    thresh_b = float(np.quantile(norm_b, percentile))
    flag_a = norm_a >= thresh_a
    flag_b = norm_b >= thresh_b
    consensus_idx = np.where(flag_a & flag_b)[0]

    print(f"\n{'=' * 64}")
    print(f"Consensus windows (top {int(percentile*100)}% in BOTH lexicons)")
    print(f"  {label_a}: {flag_a.sum()} flagged    {label_b}: {flag_b.sum()} flagged")
    print(f"  Consensus: {len(consensus_idx)} windows")
    print(f"{'=' * 64}")

    rows = []
    for rank, i in enumerate(consensus_idx, start=1):
        sent = str(df_a.loc[i, "Center_Sentence"])
        rows.append({
            "rank":             rank,
            "window_index":     int(i),
            "window_id":        int(df_a.loc[i, "Window_ID"]),
            f"nei_{label_a}":   float(nei_a[i]),
            f"nei_{label_b}":   float(nei_b[i]),
            f"norm_{label_a}":  float(norm_a[i]),
            f"norm_{label_b}":  float(norm_b[i]),
            "center_sentence":  sent,
        })
        # Truncate sentence for stdout legibility
        preview = sent if len(sent) <= 100 else sent[:97] + "..."
        print(f"  #{rank:2d}  win={int(df_a.loc[i, 'Window_ID']):4d}  "
              f"{label_a}={float(nei_a[i]):5.2f}  {label_b}={float(nei_b[i]):5.2f}  "
              f"| {preview}")
    print()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Extract consensus entrapment sentences from two-lexicon NEI series.",
    )
    ap.add_argument("--a", required=True, type=Path, help="windows CSV for lexicon A")
    ap.add_argument("--b", required=True, type=Path, help="windows CSV for lexicon B")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--method", default=config.NEI_METHOD,
                    choices=["gated_sum", "add", "mult"])
    ap.add_argument("--min-component", type=float, default=config.NEI_MIN_COMPONENT)
    ap.add_argument("--percentile", type=float, default=config.NEI_PERCENTILE)
    ap.add_argument("--out", type=Path, default=None,
                    help="optional CSV path to persist the consensus sentence list")
    args = ap.parse_args()

    df = extract_consensus(
        args.a, args.b, args.label_a, args.label_b,
        method=args.method, min_component=args.min_component, percentile=args.percentile,
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
