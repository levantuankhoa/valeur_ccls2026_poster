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

Example (sentence list only):
    python analysis_consensus_sentences.py \
        --a results/windows_warriner.csv --label-a warriner \
        --b results/windows_nrc.csv      --label-b nrc \
        --out results/consensus_sentences.csv

Example (Tier-1 episode list with per-lexicon episode lookup):
    python analysis_consensus_sentences.py \
        --a results/windows_warriner.csv --label-a warriner \
        --b results/windows_nrc.csv      --label-b nrc \
        --out          results/consensus_sentences.csv \
        --episodes-a   results/warriner.episodes.csv \
        --episodes-b   results/nrc.episodes.csv \
        --out-episodes results/consensus_episodes.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from nei_plot import compute_nei, concat_window_texts, extract_contiguous_regions

# Force UTF-8 stdout/stderr — keeps non-ASCII glyphs (—, →) from
# crashing on cp1252 Windows consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def extract_consensus(
    path_a: Path,
    path_b: Path,
    label_a: str,
    label_b: str,
    method: str = config.NEI_METHOD,
    min_component: float = config.NEI_MIN_COMPONENT,
    percentile: float = config.NEI_PERCENTILE,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Returns (sentence_rows_df, consensus_idx, aligned_df_a).

    The latter two are needed by build_consensus_episodes() to group the
    consensus windows into contiguous clusters without recomputing NEI.
    """
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

    has_full = "Full_Window_Text" in df_a.columns
    rows = []
    for rank, i in enumerate(consensus_idx, start=1):
        sent = str(df_a.loc[i, "Center_Sentence"])
        full = str(df_a.loc[i, "Full_Window_Text"]) if has_full else ""
        rows.append({
            "rank":              rank,
            "window_index":      int(i),
            "window_id":         int(df_a.loc[i, "Window_ID"]),
            f"nei_{label_a}":    float(nei_a[i]),
            f"nei_{label_b}":    float(nei_b[i]),
            f"norm_{label_a}":   float(norm_a[i]),
            f"norm_{label_b}":   float(norm_b[i]),
            "center_sentence":   sent,
            "full_window_text":  full,
        })
        # Truncate sentence for stdout legibility
        preview = sent if len(sent) <= 100 else sent[:97] + "..."
        print(f"  #{rank:2d}  win={int(df_a.loc[i, 'Window_ID']):4d}  "
              f"{label_a}={float(nei_a[i]):5.2f}  {label_b}={float(nei_b[i]):5.2f}  "
              f"| {preview}")
    print()
    return pd.DataFrame(rows), consensus_idx, df_a


def build_consensus_episodes(
    consensus_idx: np.ndarray,
    df_a: pd.DataFrame,
    label_a: str,
    label_b: str,
    episodes_a_path: Path | None = None,
    episodes_b_path: Path | None = None,
) -> pd.DataFrame:
    """Group consensus windows into contiguous clusters (strict — no gap tolerance).

    For each cluster, builds a clean paragraph by merging Full_Window_Text
    across the cluster's windows with longest-suffix-prefix overlap dedupe,
    and looks up which per-lexicon episode each cluster overlaps.
    """
    if len(consensus_idx) == 0:
        return pd.DataFrame()

    n = len(df_a)
    mask = np.zeros(n, dtype=bool)
    mask[consensus_idx] = True
    clusters = extract_contiguous_regions(mask, min_duration=1)

    eps_a = (
        pd.read_csv(episodes_a_path, encoding="utf-8-sig")
        if episodes_a_path is not None and episodes_a_path.exists()
        else None
    )
    eps_b = (
        pd.read_csv(episodes_b_path, encoding="utf-8-sig")
        if episodes_b_path is not None and episodes_b_path.exists()
        else None
    )

    def _find_overlapping_episode_ids(eps_df: pd.DataFrame | None, sw: int, ew: int) -> str:
        if eps_df is None or "episode_id" not in eps_df.columns:
            return ""
        hit = eps_df[(eps_df["start_window"] <= ew) & (eps_df["end_window"] >= sw)]
        return ",".join(str(int(x)) for x in hit["episode_id"].values) if len(hit) else ""

    has_full = "Full_Window_Text" in df_a.columns

    print(f"[consensus-episodes] {len(clusters)} contiguous cluster(s) from "
          f"{len(consensus_idx)} consensus windows (strict, no gap tolerance)")

    rows = []
    for cid, (s, e) in enumerate(clusters, start=1):
        if has_full:
            window_texts = [str(df_a.loc[k, "Full_Window_Text"]) for k in range(s, e + 1)]
            episode_full_text = concat_window_texts(window_texts)
        else:
            episode_full_text = ""

        sw = int(df_a.loc[s, "Window_ID"])
        ew = int(df_a.loc[e, "Window_ID"])

        rows.append({
            "consensus_episode_id":     cid,
            "start_window":             sw,
            "end_window":               ew,
            "duration":                 int(e - s + 1),
            f"involves_{label_a}_ep":   _find_overlapping_episode_ids(eps_a, sw, ew),
            f"involves_{label_b}_ep":   _find_overlapping_episode_ids(eps_b, sw, ew),
            "episode_full_text":        episode_full_text,
        })
        print(f"  c#{cid}  win {sw}-{ew}  dur={e - s + 1}  "
              f"({label_a}_ep={rows[-1][f'involves_{label_a}_ep'] or '-'}, "
              f"{label_b}_ep={rows[-1][f'involves_{label_b}_ep'] or '-'})")
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
    ap.add_argument("--episodes-a", type=Path, default=None,
                    help="optional path to lexicon A's *.episodes.csv (for involves_*_ep lookup)")
    ap.add_argument("--episodes-b", type=Path, default=None,
                    help="optional path to lexicon B's *.episodes.csv")
    ap.add_argument("--out-episodes", type=Path, default=None,
                    help="optional CSV path for the contiguous-cluster consensus episode list")
    args = ap.parse_args()

    df, consensus_idx, aligned_df_a = extract_consensus(
        args.a, args.b, args.label_a, args.label_b,
        method=args.method, min_component=args.min_component, percentile=args.percentile,
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"[save] {args.out}")

    if args.out_episodes is not None:
        ep_df = build_consensus_episodes(
            consensus_idx, aligned_df_a,
            args.label_a, args.label_b,
            episodes_a_path=args.episodes_a,
            episodes_b_path=args.episodes_b,
        )
        args.out_episodes.parent.mkdir(parents=True, exist_ok=True)
        ep_df.to_csv(args.out_episodes, index=False, encoding="utf-8-sig")
        print(f"[save] {args.out_episodes}")


if __name__ == "__main__":
    main()
