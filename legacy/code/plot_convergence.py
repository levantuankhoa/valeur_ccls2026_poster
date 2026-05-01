#!/usr/bin/env python3
"""
Plot VAD trajectories with convergence ROIs highlighted,
and compare ROI overlap between Warriner and NRC.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load data
war = pd.read_csv("results/new_metamorphosis_Warriner_Ridge.nei.csv", encoding="utf-8-sig")
nrc = pd.read_csv("results/new_metamorphosis_NRC_Ridge.nei.csv", encoding="utf-8-sig")

war_rois = pd.read_csv("results/new_metamorphosis_Warriner_Ridge.rois.csv", encoding="utf-8-sig")
nrc_rois = pd.read_csv("results/new_metamorphosis_NRC_Ridge.rois.csv", encoding="utf-8-sig")

# ROI overlap analysis
print("="*60)
print("  ROI OVERLAP: Warriner vs NRC")
print("="*60)

war_roi_windows = set()
for _, r in war_rois.iterrows():
    war_roi_windows |= set(range(int(r["start_window"]), int(r["end_window"]) + 1))

nrc_roi_windows = set()
for _, r in nrc_rois.iterrows():
    nrc_roi_windows |= set(range(int(r["start_window"]), int(r["end_window"]) + 1))

both_roi = war_roi_windows & nrc_roi_windows
jaccard_roi = len(both_roi) / len(war_roi_windows | nrc_roi_windows) if (war_roi_windows | nrc_roi_windows) else 0

print(f"  Warriner ROIs: {len(war_rois)} regions, {len(war_roi_windows)} windows")
print(f"  NRC ROIs:      {len(nrc_rois)} regions, {len(nrc_roi_windows)} windows")
print(f"  Overlap:       {len(both_roi)} windows")
print(f"  Jaccard:       {jaccard_roi:.3f}")

print(f"\n  Warriner ROIs:")
for _, r in war_rois.iterrows():
    s, e = int(r["start_window"]), int(r["end_window"])
    nrc_overlap = len(set(range(s, e+1)) & nrc_roi_windows)
    tag = "ALSO IN NRC" if nrc_overlap > 0 else "only Warriner"
    print(f"    W{s:3d}–{e:3d} (dur={e-s+1:2d})  range={r['mean_range']:.3f}  [{tag}]")

print(f"\n  NRC ROIs:")
for _, r in nrc_rois.iterrows():
    s, e = int(r["start_window"]), int(r["end_window"])
    war_overlap = len(set(range(s, e+1)) & war_roi_windows)
    tag = "ALSO IN WAR" if war_overlap > 0 else "only NRC"
    print(f"    W{s:3d}–{e:3d} (dur={e-s+1:2d})  range={r['mean_range']:.3f}  [{tag}]")

# =========================================================================
# PLOT: VAD trajectories with convergence zones for EACH lexicon
# =========================================================================

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

for ax, df, rois_df, lex_name, scale_label in [
    (axes[0], war, war_rois, "Warriner", "VAD Score (1-9)"),
    (axes[1], nrc, nrc_rois, "NRC", "VAD Score (-1 to +1)"),
]:
    V = df["Valence_Smooth"].values
    A = df["Arousal_Smooth"].values
    D = df["Dominance_Smooth"].values
    x = np.arange(len(V))

    # VAD range per window
    vad_stack = np.vstack([V, A, D])
    vad_range = vad_stack.max(axis=0) - vad_stack.min(axis=0)

    ymin = min(V.min(), A.min(), D.min()) - 0.3
    ymax = max(V.max(), A.max(), D.max()) + 0.3
    ax.set_ylim(ymin, ymax)

    # Plot ROIs
    for _, r in rois_df.iterrows():
        s, e = int(r["start_window"]) - 1, int(r["end_window"]) - 1  # 0-indexed
        ax.axvspan(s - 0.5, e + 0.5, facecolor="#FFD700", edgecolor="#B8860B",
                   alpha=0.3, linewidth=1.2, zorder=1)

    # Entrapment episodes
    if "Episode_ID" in df.columns:
        for eid in df["Episode_ID"].unique():
            if eid == 0:
                continue
            ep = df[df["Episode_ID"] == eid]
            s = ep.index[0]
            e = ep.index[-1]
            ax.axvspan(s - 0.5, e + 0.5, facecolor="#ff3d81", edgecolor="#ff3d81",
                       alpha=0.15, linewidth=1.5, zorder=2)

    # VAD lines
    l1, = ax.plot(x, V, label="Valence", linewidth=1.8, color="#1f77b4", zorder=10)
    l2, = ax.plot(x, A, label="Arousal", linewidth=1.8, color="#ff7f0e", zorder=10)
    l3, = ax.plot(x, D, label="Dominance", linewidth=1.8, color="#2ca02c", zorder=10)

    # Mark minimum-range points (top convergence moments)
    top_conv_idx = np.argsort(vad_range)[:5]
    for idx in top_conv_idx:
        mid_val = (V[idx] + A[idx] + D[idx]) / 3
        ax.annotate(f"W{idx+1}\nrange={vad_range[idx]:.3f}",
                    xy=(idx, mid_val), fontsize=7, ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.8),
                    arrowprops=dict(arrowstyle="->", color="red"),
                    xytext=(idx + 30, ymax - 0.2), zorder=20)

    conv_patch = mpatches.Patch(facecolor="#FFD700", edgecolor="#B8860B", alpha=0.3,
                                label="VAD Convergence (ROI)")
    entr_patch = mpatches.Patch(facecolor="#ff3d81", edgecolor="#ff3d81", alpha=0.15,
                                label="Entrapment episode")
    ax.legend(handles=[l1, l2, l3, conv_patch, entr_patch], loc="upper right", framealpha=0.95, fontsize=8)
    ax.set_ylabel(scale_label)
    ax.set_title(f"Affective Trajectory — {lex_name} Ridge (gated_sum)", fontsize=12)
    ax.grid(alpha=0.15)

axes[1].set_xlabel("Window Index (Narrative Progress)")
plt.tight_layout()
plt.savefig("results/vad_convergence_both_lexicons.png", dpi=200)
plt.close()
print(f"\n[saved] results/vad_convergence_both_lexicons.png")

# =========================================================================
# PLOT 2: VAD range over narrative — shows WHERE the 3 lines come closest
# =========================================================================
fig2, axes2 = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

for ax, df, rois_df, lex_name in [
    (axes2[0], war, war_rois, "Warriner"),
    (axes2[1], nrc, nrc_rois, "NRC"),
]:
    V = df["Valence_Smooth"].values
    A = df["Arousal_Smooth"].values
    D = df["Dominance_Smooth"].values
    vad_stack = np.vstack([V, A, D])
    vad_range = vad_stack.max(axis=0) - vad_stack.min(axis=0)
    x = np.arange(len(vad_range))

    ax.fill_between(x, vad_range, alpha=0.4, color="#2196F3")
    ax.plot(x, vad_range, linewidth=1, color="#1565C0")

    # Convergence threshold
    import config
    thresh_key = "warriner" if lex_name == "Warriner" else "nrc"
    thresh = config.CONVERGENCE_THRESHOLD[thresh_key]
    ax.axhline(thresh, color="red", linestyle="--", linewidth=1.5, label=f"Threshold = {thresh}")

    # Highlight ROIs
    for _, r in rois_df.iterrows():
        s, e = int(r["start_window"]) - 1, int(r["end_window"]) - 1
        ax.axvspan(s - 0.5, e + 0.5, facecolor="#FFD700", alpha=0.3, zorder=1)

    ax.set_ylabel("VAD Range (max - min)")
    ax.set_title(f"VAD Range — {lex_name} (lower = more convergence)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

axes2[1].set_xlabel("Window Index (Narrative Progress)")
plt.tight_layout()
plt.savefig("results/vad_range_convergence.png", dpi=200)
plt.close()
print(f"[saved] results/vad_range_convergence.png")

# =========================================================================
# PLOT 3: Consensus entrapment + convergence overlay (both lexicons on one)
# =========================================================================
fig3, ax3 = plt.subplots(figsize=(16, 5))

# Normalize both to 0-1 for comparison
war_nei_norm = war["NEI"].values / (war["NEI"].max() + 1e-9)
nrc_nei_norm = nrc["NEI"].values / (nrc["NEI"].max() + 1e-9)
x = np.arange(len(war_nei_norm))

ax3.fill_between(x, war_nei_norm, alpha=0.3, color="#1f77b4", label="Warriner NEI (normalized)")
ax3.fill_between(x, nrc_nei_norm, alpha=0.3, color="#ff7f0e", label="NRC NEI (normalized)")

# Mark consensus windows
consensus = set()
war_entr = set(war[war["Entrapment"] == 1]["Window_ID"].values)
nrc_entr = set(nrc[nrc["Entrapment"] == 1]["Window_ID"].values)
consensus = war_entr & nrc_entr

for w in consensus:
    ax3.axvline(w - 1, color="red", alpha=0.6, linewidth=1.5, zorder=5)

ax3.axhline(0, color="black", linewidth=0.5)
consensus_patch = plt.Line2D([0], [0], color="red", linewidth=2, label=f"Consensus ({len(consensus)} windows)")
ax3.legend(handles=[
    mpatches.Patch(facecolor="#1f77b4", alpha=0.3, label="Warriner NEI"),
    mpatches.Patch(facecolor="#ff7f0e", alpha=0.3, label="NRC NEI"),
    consensus_patch
], fontsize=9)
ax3.set_xlabel("Window Index (Narrative Progress)")
ax3.set_ylabel("NEI (normalized 0-1)")
ax3.set_title("Cross-Lexicon NEI Consensus (gated_sum)", fontsize=12)
ax3.grid(alpha=0.15)
plt.tight_layout()
plt.savefig("results/nei_consensus_overlay.png", dpi=200)
plt.close()
print(f"[saved] results/nei_consensus_overlay.png")
