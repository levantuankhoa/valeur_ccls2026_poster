#!/usr/bin/env python3
"""
Compare current Ridge results with February 2025 results.
Produces comparison plots and correlation statistics.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path

RESULTS = Path("results")

COMPARISONS = [
    {
        "label": "Warriner Ridge",
        "old_file": RESULTS / "metamorphosis_Warriner_Ridge_vad_with_rois.csv",
        "new_file": RESULTS / "new_metamorphosis_Warriner_Ridge_windows.csv",
        "old_cols": {"V": "Valence_Ridge_Smooth", "A": "Arousal_Ridge_Smooth", "D": "Dominance_Ridge_Smooth"},
        "new_cols": {"V": "Valence_Smooth", "A": "Arousal_Smooth", "D": "Dominance_Smooth"},
    },
    {
        "label": "NRC Ridge",
        "old_file": RESULTS / "metamorphosis_NRC_Ridge_vad_with_rois.csv",
        "new_file": RESULTS / "new_metamorphosis_NRC_Ridge_windows.csv",
        "old_cols": {"V": "Valence_Ridge_Smooth", "A": "Arousal_Ridge_Smooth", "D": "Dominance_Ridge_Smooth"},
        "new_cols": {"V": "Valence_Smooth", "A": "Arousal_Smooth", "D": "Dominance_Smooth"},
    },
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("So sánh kết quả Ridge: Hiện tại vs Tháng 2/2025", fontsize=14, fontweight="bold")

dims = ["V", "A", "D"]
dim_names = ["Valence", "Arousal", "Dominance"]
colors = {"V": "#1f77b4", "A": "#ff7f0e", "D": "#2ca02c"}

for row_idx, comp in enumerate(COMPARISONS):
    old_df = pd.read_csv(comp["old_file"], encoding="utf-8-sig")
    new_df = pd.read_csv(comp["new_file"], encoding="utf-8-sig")

    # Align length
    n = min(len(old_df), len(new_df))
    x = np.linspace(0, 1, n)

    print(f"\n{'='*60}")
    print(f"  {comp['label']}  (N={n} windows)")
    print(f"{'='*60}")

    for col_idx, (dim, dim_name) in enumerate(zip(dims, dim_names)):
        ax = axes[row_idx, col_idx]
        old_vals = old_df[comp["old_cols"][dim]].values[:n]
        new_vals = new_df[comp["new_cols"][dim]].values[:n]

        # Statistics
        r, p = pearsonr(old_vals, new_vals)
        rmse = np.sqrt(np.mean((old_vals - new_vals) ** 2))
        mae = np.mean(np.abs(old_vals - new_vals))
        max_diff = np.max(np.abs(old_vals - new_vals))

        print(f"  {dim_name:10s}  r={r:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  MaxDiff={max_diff:.4f}")

        # Plot overlay
        ax.plot(x, old_vals, label="Feb 2025", color=colors[dim], alpha=0.7, linewidth=1.5)
        ax.plot(x, new_vals, label="Current", color=colors[dim], linestyle="--", alpha=0.9, linewidth=1.5)

        # Difference shading
        ax.fill_between(x, old_vals, new_vals, alpha=0.15, color="red")

        ax.set_title(f"{comp['label']} — {dim_name}\nr={r:.4f}  RMSE={rmse:.4f}", fontsize=10)
        ax.set_xlabel("Narrative Progress")
        ax.set_ylabel(dim_name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

plt.tight_layout()
out_path = RESULTS / "comparison_old_vs_new_ridge.png"
plt.savefig(out_path, dpi=200)
plt.close()
print(f"\n[saved] {out_path}")

# --- Scatter plots: old vs new ---
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle("Scatter: Kết quả hiện tại vs Tháng 2/2025 (mỗi điểm = 1 window)", fontsize=14, fontweight="bold")

for row_idx, comp in enumerate(COMPARISONS):
    old_df = pd.read_csv(comp["old_file"], encoding="utf-8-sig")
    new_df = pd.read_csv(comp["new_file"], encoding="utf-8-sig")
    n = min(len(old_df), len(new_df))

    for col_idx, (dim, dim_name) in enumerate(zip(dims, dim_names)):
        ax = axes2[row_idx, col_idx]
        old_vals = old_df[comp["old_cols"][dim]].values[:n]
        new_vals = new_df[comp["new_cols"][dim]].values[:n]

        r, _ = pearsonr(old_vals, new_vals)
        ax.scatter(old_vals, new_vals, s=5, alpha=0.4, color=colors[dim])

        # Perfect agreement line
        vmin = min(old_vals.min(), new_vals.min())
        vmax = max(old_vals.max(), new_vals.max())
        ax.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.5, linewidth=1)

        ax.set_title(f"{comp['label']} — {dim_name}  (r={r:.4f})", fontsize=10)
        ax.set_xlabel("Feb 2025")
        ax.set_ylabel("Current")
        ax.grid(alpha=0.2)

plt.tight_layout()
out_path2 = RESULTS / "scatter_old_vs_new_ridge.png"
plt.savefig(out_path2, dpi=200)
plt.close()
print(f"[saved] {out_path2}")

# --- Difference distribution ---
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
fig3.suptitle("Phân phối sai lệch (Current - Feb 2025)", fontsize=14, fontweight="bold")

for row_idx, comp in enumerate(COMPARISONS):
    old_df = pd.read_csv(comp["old_file"], encoding="utf-8-sig")
    new_df = pd.read_csv(comp["new_file"], encoding="utf-8-sig")
    n = min(len(old_df), len(new_df))

    for col_idx, (dim, dim_name) in enumerate(zip(dims, dim_names)):
        ax = axes3[row_idx, col_idx]
        old_vals = old_df[comp["old_cols"][dim]].values[:n]
        new_vals = new_df[comp["new_cols"][dim]].values[:n]
        diff = new_vals - old_vals

        ax.hist(diff, bins=50, color=colors[dim], alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.axvline(diff.mean(), color="black", linestyle="-", linewidth=1.5, label=f"mean={diff.mean():.4f}")
        ax.set_title(f"{comp['label']} — {dim_name}\nmean={diff.mean():.4f}  std={diff.std():.4f}", fontsize=10)
        ax.set_xlabel("Difference (Current - Feb 2025)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

plt.tight_layout()
out_path3 = RESULTS / "diff_distribution_old_vs_new_ridge.png"
plt.savefig(out_path3, dpi=200)
plt.close()
print(f"[saved] {out_path3}")
