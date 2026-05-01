#!/usr/bin/env python3
"""
Valeur — Step 3: compute the Narrative Entrapment Index (NEI),
extract entrapment episodes and convergence ROIs, render trajectory plot.

The NEI operationalises Gilbert & Allan's (1998) clinical entrapment
state — simultaneous Valence↓, Arousal↑, Dominance↓ — on an intra-text
z-score basis. The default `gated_sum` method requires all three
z-components to exceed MIN_COMPONENT before contributing, eliminating
the false-positive failure mode of the naive additive formulation on
lexicons with near-independent affective channels (e.g., NRC VAD v2.1).

Example (single lexicon):
    python nei_plot.py --windows results/windows_warriner.csv \
        --lexicon warriner --out-prefix results/warriner

Example (dual-lexicon consensus heatmap):
    python nei_plot.py --consensus \
        --windows results/windows_warriner.csv results/windows_nrc.csv \
        --lexicon warriner nrc \
        --out-prefix results/consensus
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


# =============================================================================
# NEI computation
# =============================================================================
def compute_nei(
    V: np.ndarray,
    A: np.ndarray,
    D: np.ndarray,
    method: str = config.NEI_METHOD,
    min_component: float = config.NEI_MIN_COMPONENT,
    baseline: str = config.BASELINE_METHOD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Narrative Entrapment Index from smoothed V/A/D trajectories.

    Returns
    -------
    entr_index : np.ndarray (N,)        NEI per window
    zV_drop    : np.ndarray (N,)        rectified z-score of valence drop
    zA_rise    : np.ndarray (N,)        rectified z-score of arousal rise
    zD_drop    : np.ndarray (N,)        rectified z-score of dominance drop
    """
    if baseline == "median":
        baseV, baseA, baseD = np.median(V), np.median(A), np.median(D)
    elif baseline == "mean":
        baseV, baseA, baseD = np.mean(V), np.mean(A), np.mean(D)
    else:
        raise ValueError(f"baseline must be 'median' or 'mean', got {baseline!r}")

    # Directional deltas: positive values indicate movement toward entrapment
    v_drop = baseV - V
    a_rise = A - baseA
    d_drop = baseD - D

    eps = 1e-9
    zV = np.maximum((v_drop - v_drop.mean()) / (v_drop.std() + eps), 0.0)
    zA = np.maximum((a_rise - a_rise.mean()) / (a_rise.std() + eps), 0.0)
    zD = np.maximum((d_drop - d_drop.mean()) / (d_drop.std() + eps), 0.0)

    if method == "add":
        # Naive additive — known to yield ~74% false positives on
        # near-independent lexicons (NRC). Kept for ablation.
        nei = zV + zA + zD
    elif method == "mult":
        # Multiplicative — vanishes when any component is zero, but noisy.
        nei = zV * zA * zD
    elif method == "gated_sum":
        # Gated sum: require all three channels to clear MIN_COMPONENT.
        # This enforces Mehrabian–Russell's joint-condition semantics.
        gate = (zV > min_component) & (zA > min_component) & (zD > min_component)
        nei = np.where(gate, zV + zA + zD, 0.0)
    else:
        raise ValueError(f"unknown NEI method: {method!r}")

    return nei, zV, zA, zD


def diagnose_components(zV: np.ndarray, zA: np.ndarray, zD: np.ndarray, min_comp: float) -> None:
    """Print fraction of windows where each component exceeds the gate — useful for audit."""
    n = len(zV)
    frac_V = (zV > min_comp).mean() * 100
    frac_A = (zA > min_comp).mean() * 100
    frac_D = (zD > min_comp).mean() * 100
    frac_all = ((zV > min_comp) & (zA > min_comp) & (zD > min_comp)).mean() * 100
    print(f"[diagnose] gate={min_comp}")
    print(f"           zV>gate: {frac_V:5.1f}%   zA>gate: {frac_A:5.1f}%   zD>gate: {frac_D:5.1f}%")
    print(f"           all three:  {frac_all:5.1f}%   (N={n})")


# =============================================================================
# Episode & ROI extraction
# =============================================================================
def extract_contiguous_regions(mask: np.ndarray, min_duration: int = 1) -> List[Tuple[int, int]]:
    """Return list of (start, end) inclusive indices where mask is True."""
    regions = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if (j - i + 1) >= min_duration:
                regions.append((i, j))
            i = j + 1
        else:
            i += 1
    return regions


# =============================================================================
# Plotting
# =============================================================================
def plot_trajectory(
    df: pd.DataFrame,
    V: np.ndarray, A: np.ndarray, D: np.ndarray,
    rois: List[Tuple[int, int]],
    episodes: List[Tuple[int, int]],
    title: str,
    out_path: Path,
    y_margin: float = 0.6,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(V))

    ymin = min(V.min(), A.min(), D.min()) - y_margin
    ymax = max(V.max(), A.max(), D.max()) + y_margin
    ax.set_ylim(ymin, ymax)

    l1, = ax.plot(x, V, label="Valence",   linewidth=1.8, color="#1f77b4", zorder=10)
    l2, = ax.plot(x, A, label="Arousal",   linewidth=1.8, color="#ff7f0e", zorder=10)
    l3, = ax.plot(x, D, label="Dominance", linewidth=1.8, color="#2ca02c", zorder=10)

    for (s, e) in rois:
        ax.add_patch(mpatches.Rectangle(
            (s - 0.5, ymin), e - s + 1, ymax - ymin,
            facecolor="#9e9e9e", edgecolor="#4f4f4f", linewidth=1.4, alpha=0.22, zorder=1,
        ))
    for (s, e) in episodes:
        ax.add_patch(mpatches.Rectangle(
            (s - 0.5, ymin), e - s + 1, ymax - ymin,
            facecolor="#ffb3d9", edgecolor="#ff3d81", linewidth=2.0, alpha=0.30, zorder=5,
        ))

    conv_patch = mpatches.Patch(facecolor="#9e9e9e", edgecolor="#4f4f4f", alpha=0.22,
                                label="VAD convergence (ROI)")
    entr_patch = mpatches.Patch(facecolor="#ffb3d9", edgecolor="#ff3d81", alpha=0.30,
                                label="Entrapment episode")
    ax.legend(handles=[l1, l2, l3, conv_patch, entr_patch], loc="upper left", framealpha=0.95)
    ax.set_xlabel("Window index (narrative progress →)")
    ax.set_ylabel("V / A / D score")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot] {out_path}")


# =============================================================================
# Single-lexicon pipeline
# =============================================================================
def run_single(
    windows_path: Path,
    lexicon: str,
    out_prefix: Path,
    method: str = config.NEI_METHOD,
    min_component: float = config.NEI_MIN_COMPONENT,
    percentile: float = config.NEI_PERCENTILE,
) -> pd.DataFrame:
    print(f"\n{'=' * 60}")
    print(f"▶ NEI ({lexicon}, method={method})")
    print(f"{'=' * 60}")

    df = pd.read_csv(windows_path, encoding="utf-8-sig")
    V = df["Valence_Smooth"].astype(float).values
    A = df["Arousal_Smooth"].astype(float).values
    D = df["Dominance_Smooth"].astype(float).values

    # --- NEI ---
    nei, zV, zA, zD = compute_nei(V, A, D, method=method, min_component=min_component)
    diagnose_components(zV, zA, zD, min_component)

    cutoff   = float(np.quantile(nei, percentile))
    entr_mask = nei >= cutoff
    df["NEI"]        = nei
    df["zV_drop"]    = zV
    df["zA_rise"]    = zA
    df["zD_drop"]    = zD
    df["Entrapment"] = entr_mask.astype(int)
    print(f"[nei] cutoff (p{int(percentile*100)})={cutoff:.3f}   flagged={entr_mask.sum()}/{len(df)}")

    # --- Convergence ROIs ---
    vad_stack = np.vstack([V, A, D])
    vad_range = vad_stack.max(axis=0) - vad_stack.min(axis=0)
    conv_thresh = config.CONVERGENCE_THRESHOLD[lexicon]
    conv_mask = vad_range <= conv_thresh
    df["VAD_range"]   = vad_range
    df["Convergence"] = conv_mask.astype(int)

    rois     = extract_contiguous_regions(conv_mask,  min_duration=config.MIN_EPISODE_DURATION)
    episodes = extract_contiguous_regions(entr_mask, min_duration=config.MIN_EPISODE_DURATION)
    print(f"[regions] {len(rois)} convergence ROIs  |  {len(episodes)} entrapment episodes")

    # Assign episode IDs back into df
    df["Episode_ID"] = 0
    for eid, (s, e) in enumerate(episodes, start=1):
        df.loc[s:e, "Episode_ID"] = eid

    # --- Persist ---
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_prefix.with_suffix(".nei.csv"), index=False, encoding="utf-8-sig")

    ep_rows = []
    for eid, (s, e) in enumerate(episodes, start=1):
        center = (s + e) // 2
        ep_rows.append({
            "episode_id":     eid,
            "start_window":   int(df.loc[s, "Window_ID"]),
            "end_window":     int(df.loc[e, "Window_ID"]),
            "duration":       int(e - s + 1),
            "max_nei":        float(nei[s:e + 1].max()),
            "mean_nei":       float(nei[s:e + 1].mean()),
            "sample_sentence": str(df.loc[center, "Center_Sentence"]) if "Center_Sentence" in df.columns else "",
        })
    if ep_rows:
        pd.DataFrame(ep_rows).to_csv(out_prefix.with_suffix(".episodes.csv"), index=False, encoding="utf-8-sig")

    roi_rows = []
    for rid, (s, e) in enumerate(rois, start=1):
        roi_rows.append({
            "roi_id":        rid,
            "start_window":  int(df.loc[s, "Window_ID"]),
            "end_window":    int(df.loc[e, "Window_ID"]),
            "duration":      int(e - s + 1),
            "mean_range":    float(vad_range[s:e + 1].mean()),
            "mean_valence":  float(V[s:e + 1].mean()),
            "mean_arousal":  float(A[s:e + 1].mean()),
            "mean_dominance": float(D[s:e + 1].mean()),
        })
    if roi_rows:
        pd.DataFrame(roi_rows).to_csv(out_prefix.with_suffix(".rois.csv"), index=False, encoding="utf-8-sig")

    # --- Plot ---
    y_margin = 0.6 if lexicon == "warriner" else 0.15
    plot_trajectory(
        df, V, A, D, rois, episodes,
        title=f"Affective trajectory — {lexicon.capitalize()} ({method})",
        out_path=out_prefix.with_suffix(".plot.png"),
        y_margin=y_margin,
    )
    return df


# =============================================================================
# Dual-lexicon consensus
# =============================================================================
def run_consensus(
    window_paths: List[Path],
    lexicon_names: List[str],
    out_prefix: Path,
    method: str = config.NEI_METHOD,
    min_component: float = config.NEI_MIN_COMPONENT,
    percentile: float = config.NEI_PERCENTILE,
) -> None:
    """Align NEI across lexicons, compute consensus, render heatmap."""
    from scipy.stats import spearmanr

    if len(window_paths) != len(lexicon_names):
        raise ValueError("--windows and --lexicon must have the same number of arguments")

    series = {}
    for path, name in zip(window_paths, lexicon_names):
        df = pd.read_csv(path, encoding="utf-8-sig")
        V = df["Valence_Smooth"].values
        A = df["Arousal_Smooth"].values
        D = df["Dominance_Smooth"].values
        nei, *_ = compute_nei(V, A, D, method=method, min_component=min_component)
        series[name] = nei

    # Align on min length (different windowing edge cases)
    L = min(len(s) for s in series.values())
    mat = np.stack([series[n][:L] for n in lexicon_names])

    # Cross-lexicon Spearman
    print(f"\n[consensus] cross-lexicon Spearman rho on NEI:")
    for i in range(len(lexicon_names)):
        for j in range(i + 1, len(lexicon_names)):
            rho, p = spearmanr(mat[i], mat[j])
            print(f"  {lexicon_names[i]} vs {lexicon_names[j]}:  rho={rho:.3f}  p={p:.3e}")

    # Row-wise min-max normalize for visualization
    norm = (mat - mat.min(axis=1, keepdims=True)) / (mat.ptp(axis=1, keepdims=True) + 1e-9)

    # Consensus: windows flagged in all lexicons at top-percentile
    thresholds = np.quantile(norm, percentile, axis=1, keepdims=True)
    flags = (norm >= thresholds).astype(int)
    votes = flags.sum(axis=0)
    consensus_windows = np.where(votes == len(lexicon_names))[0]
    print(f"[consensus] windows flagged by ALL {len(lexicon_names)} lexicons: {len(consensus_windows)}")

    # Save
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mat.T, columns=lexicon_names).to_csv(
        out_prefix.with_suffix(".nei_matrix.csv"), index_label="Window_ID",
    )

    # Heatmap
    fig, ax = plt.subplots(figsize=(16, 3 + 0.5 * len(lexicon_names)))
    vmax = np.percentile(norm, 98)
    im = ax.imshow(norm, aspect="auto", cmap="Reds", vmin=0, vmax=vmax)
    ax.set_yticks(range(len(lexicon_names)))
    ax.set_yticklabels(lexicon_names)
    ax.set_xlabel("Window index (narrative progress →)")
    ax.set_title(f"Cross-lexicon NEI ({method})  —  consensus windows: {len(consensus_windows)}")
    fig.colorbar(im, ax=ax, label="NEI (row-normalized)")
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".heatmap.png"), dpi=300)
    plt.close()
    print(f"[plot] {out_prefix.with_suffix('.heatmap.png')}")


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="NEI + plot (Valeur step 3/3)")
    ap.add_argument("--windows", required=True, nargs="+", type=Path,
                    help="one or more VAD windows CSVs (from encode_vad.py)")
    ap.add_argument("--lexicon", required=True, nargs="+", choices=list(config.LEXICONS.keys()),
                    help="lexicon name(s) — must match order of --windows")
    ap.add_argument("--out-prefix", required=True, type=Path,
                    help="output path prefix (suffixes .nei.csv, .episodes.csv, .plot.png added)")
    ap.add_argument("--method", default=config.NEI_METHOD,
                    choices=["gated_sum", "add", "mult"])
    ap.add_argument("--min-component", type=float, default=config.NEI_MIN_COMPONENT,
                    help="z-gate threshold for gated_sum method")
    ap.add_argument("--percentile", type=float, default=config.NEI_PERCENTILE)
    ap.add_argument("--consensus", action="store_true",
                    help="run dual-lexicon consensus mode (requires ≥2 --windows)")
    args = ap.parse_args()

    if args.consensus or len(args.windows) > 1:
        run_consensus(
            args.windows, args.lexicon, args.out_prefix,
            method=args.method, min_component=args.min_component, percentile=args.percentile,
        )
    else:
        run_single(
            args.windows[0], args.lexicon[0], args.out_prefix,
            method=args.method, min_component=args.min_component, percentile=args.percentile,
        )


if __name__ == "__main__":
    main()
