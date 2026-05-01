#!/usr/bin/env python3
"""Diagnose which NEI method the old Feb 2025 results used."""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Load old results (Feb 2025)
old_nrc = pd.read_csv("results/metamorphosis_NRC_Ridge_vad_with_rois.csv", encoding="utf-8-sig")
old_war = pd.read_csv("results/metamorphosis_Warriner_Ridge_vad_with_rois.csv", encoding="utf-8-sig")

# Load new windows (current pipeline)
new_nrc = pd.read_csv("results/new_metamorphosis_NRC_Ridge_windows.csv", encoding="utf-8-sig")
new_war = pd.read_csv("results/new_metamorphosis_Warriner_Ridge_windows.csv", encoding="utf-8-sig")

def compute_nei_all_methods(V, A, D, baseline="median", min_component=0.10):
    """Compute NEI using all 3 methods, return dict of arrays."""
    if baseline == "median":
        baseV, baseA, baseD = np.median(V), np.median(A), np.median(D)
    else:
        baseV, baseA, baseD = np.mean(V), np.mean(A), np.mean(D)

    v_drop = baseV - V
    a_rise = A - baseA
    d_drop = baseD - D

    eps = 1e-9
    zV = np.maximum((v_drop - v_drop.mean()) / (v_drop.std() + eps), 0.0)
    zA = np.maximum((a_rise - a_rise.mean()) / (a_rise.std() + eps), 0.0)
    zD = np.maximum((d_drop - d_drop.mean()) / (d_drop.std() + eps), 0.0)

    nei_add = zV + zA + zD
    nei_mult = zV * zA * zD
    gate = (zV > min_component) & (zA > min_component) & (zD > min_component)
    nei_gated = np.where(gate, zV + zA + zD, 0.0)

    return {"add": nei_add, "mult": nei_mult, "gated_sum": nei_gated, "zV": zV, "zA": zA, "zD": zD}


for name, old_df, new_df in [
    ("NRC Ridge", old_nrc, new_nrc),
    ("Warriner Ridge", old_war, new_war),
]:
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    # Get old Entr_Index
    old_entr = old_df["Entr_Index"].values
    n = min(len(old_entr), len(new_df))

    # Use OLD smoothed values to compute NEI (to isolate method difference from VAD difference)
    if "NRC" in name:
        V_old = old_df["Valence_Ridge_Smooth"].values[:n]
        A_old = old_df["Arousal_Ridge_Smooth"].values[:n]
        D_old = old_df["Dominance_Ridge_Smooth"].values[:n]
    else:
        V_old = old_df["Valence_Ridge_Smooth"].values[:n]
        A_old = old_df["Arousal_Ridge_Smooth"].values[:n]
        D_old = old_df["Dominance_Ridge_Smooth"].values[:n]

    # Compute NEI from OLD VAD values using all methods
    results_old_vad = compute_nei_all_methods(V_old, A_old, D_old)

    print(f"\n  Correlating old Entr_Index with NEI computed from OLD VAD values:")
    for method in ["add", "mult", "gated_sum"]:
        nei = results_old_vad[method][:n]
        r, p = pearsonr(old_entr[:n], nei)
        rmse = np.sqrt(np.mean((old_entr[:n] - nei) ** 2))
        # Check if values are proportional (same shape but different scale)
        if nei.std() > 0:
            scale = old_entr[:n].std() / nei.std()
        else:
            scale = float('nan')
        print(f"    {method:12s}  r={r:.6f}  RMSE={rmse:.4f}  scale_ratio={scale:.4f}")

    # Also try with mean baseline
    results_mean = compute_nei_all_methods(V_old, A_old, D_old, baseline="mean")
    print(f"\n  Same but with baseline='mean':")
    for method in ["add", "mult", "gated_sum"]:
        nei = results_mean[method][:n]
        r, p = pearsonr(old_entr[:n], nei)
        rmse = np.sqrt(np.mean((old_entr[:n] - nei) ** 2))
        print(f"    {method:12s}  r={r:.6f}  RMSE={rmse:.4f}")

    # Show first 10 windows comparison
    print(f"\n  First 10 windows — old Entr_Index vs methods (from OLD VAD):")
    print(f"  {'Win':>4s}  {'Old_Entr':>10s}  {'add':>10s}  {'gated':>10s}  {'mult':>10s}")
    for i in range(min(10, n)):
        print(f"  {i+1:4d}  {old_entr[i]:10.4f}  {results_old_vad['add'][i]:10.4f}  "
              f"{results_old_vad['gated_sum'][i]:10.4f}  {results_old_vad['mult'][i]:10.4f}")

    # Component analysis at opening
    print(f"\n  z-components at opening (windows 1-7) from OLD VAD:")
    print(f"  {'Win':>4s}  {'zV_drop':>8s}  {'zA_rise':>8s}  {'zD_drop':>8s}  {'all>0.1?':>8s}")
    for i in range(min(7, n)):
        zv = results_old_vad["zV"][i]
        za = results_old_vad["zA"][i]
        zd = results_old_vad["zD"][i]
        gate = "YES" if (zv > 0.1 and za > 0.1 and zd > 0.1) else "NO"
        print(f"  {i+1:4d}  {zv:8.4f}  {za:8.4f}  {zd:8.4f}  {gate:>8s}")

    # Same for NEW VAD
    V_new = new_df["Valence_Smooth"].values[:n]
    A_new = new_df["Arousal_Smooth"].values[:n]
    D_new = new_df["Dominance_Smooth"].values[:n]
    results_new_vad = compute_nei_all_methods(V_new, A_new, D_new)

    print(f"\n  z-components at opening (windows 1-7) from NEW VAD:")
    print(f"  {'Win':>4s}  {'zV_drop':>8s}  {'zA_rise':>8s}  {'zD_drop':>8s}  {'all>0.1?':>8s}")
    for i in range(min(7, n)):
        zv = results_new_vad["zV"][i]
        za = results_new_vad["zA"][i]
        zd = results_new_vad["zD"][i]
        gate = "YES" if (zv > 0.1 and za > 0.1 and zd > 0.1) else "NO"
        print(f"  {i+1:4d}  {zv:8.4f}  {za:8.4f}  {zd:8.4f}  {gate:>8s}")
