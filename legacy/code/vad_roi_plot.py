#!/usr/bin/env python3
"""
vad_roi_plot_batch.py
Automated batch processing for VAD (Valence-Arousal-Dominance) trajectories.

This script parses the output of the Dual-Lexicon (Warriner/NRC) & Dual-Model (Ridge/RF) pipeline.
For each condition, it computes Convergence ROIs (Regions of Interest) and Entrapment Episodes,
then exports a comprehensive set of 5 artifacts:
  1. Extended CSV with ROI and Entrapment flags.
  2. ROI Summary CSV.
  3. Entrapment Episodes Summary CSV.
  4. Flat windows CSV (for qualitative context extraction).
  5. High-resolution trajectory plot (.png).

Usage: python vad_roi_plot_batch.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# --------- CONFIGURATION ----------
USE_SMOOTHED = True
MIN_DURATION = 2
ENTR_METHOD = "add"      # Options: "add" or "mult"
ENTR_PERCENTILE = 0.95   # Top 5% threshold for narrative entrapment
BASELINE_METHOD = "median"  # Options: "mean" or "median"

# Input files configuration and scale-specific thresholds
FILES_CONFIG = {
    "Warriner": {
        "path": "metamorphosis_vad_warriner_windows.csv",
        "range_thresh": 0.5,  # Appropriate for 1-9 scale
        "y_margin": 0.6
    },
    "NRC": {
        "path": "metamorphosis_vad_nrc_windows.csv",
        "range_thresh": 0.15, # Appropriate for normalized 0-1 scale
        "y_margin": 0.15
    }
}
MODELS = ["Ridge", "RF"]
# ----------------------------------

def process_condition(df_main, lex_name, model_name, config):
    """
    Core function to process a specific Lexicon + Model combination.
    Calculates metrics, groups episodes, and exports all required artifacts.
    """
    print(f"\n{'='*50}")
    print(f"▶ PROCESSING: {lex_name} Lexicon | {model_name} Regressor")
    print(f"{'='*50}")
    
    df = df_main.copy()
    has_center = "Center_Sentence" in df.columns
    n = len(df)
    x = np.arange(n)
    
    # Prefix for all output artifacts
    prefix = f"metamorphosis_{lex_name}_{model_name}"
    
    # 1. Identify target columns
    val_col = f"Valence_{model_name}_Smooth" if USE_SMOOTHED else f"Valence_{model_name}"
    aro_col = f"Arousal_{model_name}_Smooth" if USE_SMOOTHED else f"Arousal_{model_name}"
    dom_col = f"Dominance_{model_name}_Smooth" if USE_SMOOTHED else f"Dominance_{model_name}"

    if dom_col not in df.columns:
        print(f"[ERROR] Required column '{dom_col}' not found in {config['path']}. Skipping...")
        return

    V = df[val_col].astype(float).values
    A = df[aro_col].astype(float).values
    D = df[dom_col].astype(float).values

    # ============================================================
    # 1️⃣  CONVERGENCE (Range Metric)
    # ============================================================
    ranges = np.max(np.vstack([V, A, D]), axis=0) - np.min(np.vstack([V, A, D]), axis=0)
    converge_mask = ranges <= config["range_thresh"]

    df["VAD_range"] = ranges
    df["Convergence"] = converge_mask.astype(int)

    # Extract continuous ROIs based on minimum duration
    rois = []
    i = 0
    while i < n:
        if converge_mask[i]:
            j = i
            while j+1 < n and converge_mask[j+1]: j += 1
            if (j - i + 1) >= MIN_DURATION:
                rois.append((i, j))
            i = j + 1
        else:
            i += 1

    # ============================================================
    # 2️⃣  ENTRAPMENT INDEX (Directional + Z-Score Normalization)
    # ============================================================
    if BASELINE_METHOD == "median":
        baseV, baseA, baseD = np.median(V), np.median(A), np.median(D)
    else:
        baseV, baseA, baseD = np.mean(V), np.mean(A), np.mean(D)
    
    # Calculate directional deltas (focused on the 'entrapment' psychological state)
    v_drop = baseV - V  # Positive when valence is lower than baseline
    a_rise = A - baseA  # Positive when arousal is higher than baseline
    d_drop = baseD - D  # Positive when dominance is lower than baseline

    eps = 1e-9
    # Apply positive-only Z-score transformation
    zV = np.maximum((v_drop - np.mean(v_drop)) / (np.std(v_drop) + eps), 0)
    zA = np.maximum((a_rise - np.mean(a_rise)) / (np.std(a_rise) + eps), 0)
    zD = np.maximum((d_drop - np.mean(d_drop)) / (np.std(d_drop) + eps), 0)

    # Compute composite index
    entr_index = zV * zA * zD if ENTR_METHOD == "mult" else zV + zA + zD
    cutoff = np.quantile(entr_index, ENTR_PERCENTILE)
    entr_mask = entr_index >= cutoff
    
    df["Entr_Index"] = entr_index
    df["Entrapment"] = entr_mask.astype(int)

    # Group continuous entrapment windows into episodes
    episodes = []
    episode_id = 1
    df["Episode_ID"] = 0

    i = 0
    while i < n:
        if entr_mask[i]:
            j = i
            while j+1 < n and entr_mask[j+1]: j += 1
            df.loc[i:j, "Episode_ID"] = episode_id
            episodes.append({
                "episode_id": episode_id,
                "start_idx": int(i), "end_idx": int(j),
                "start_window": int(df.loc[i, "Window_ID"]), "end_window": int(df.loc[j, "Window_ID"]),
                "duration": int(j - i + 1),
                "max_entr": float(np.max(entr_index[i:j+1])),
                "mean_entr": float(np.mean(entr_index[i:j+1]))
            })
            episode_id += 1
            i = j + 1
        else:
            i += 1

    # Export Extended CSV
    out_ext = f"{prefix}_vad_with_rois.csv"
    df.to_csv(out_ext, index=False, encoding="utf-8-sig")
    print(f"✔ Exported Extended Dataset -> {out_ext}")

    # ============================================================
    # 3️⃣  EXPORT ROI & ENTRAPMENT SUMMARIES (For Qualitative Analysis)
    # ============================================================
    
    # 3a. ROI Summary
    rois_rows = []
    for idx, (s,e) in enumerate(rois, start=1):
        sent_preview = None
        if has_center:
            sample_idxs = [s, (s+e)//2, e]
            sents = [str(df.loc[si, "Center_Sentence"]) for si in sample_idxs if 0 <= si < n]
            sent_preview = " || ".join(sents)
        rois_rows.append({
            "ROI_id": idx, "start_idx": int(s), "end_idx": int(e),
            "start_window": int(df.loc[s, "Window_ID"]), "end_window": int(df.loc[e, "Window_ID"]),
            "duration": int(e - s + 1),
            "mean_range": float(np.mean(ranges[s:e+1])),
            "mean_valence": float(np.mean(V[s:e+1])),
            "mean_arousal": float(np.mean(A[s:e+1])),
            "mean_dominance": float(np.mean(D[s:e+1])),
            "sample_sentences": sent_preview
        })

    if rois_rows:
        pd.DataFrame(rois_rows).to_csv(f"{prefix}_rois_summary.csv", index=False, encoding="utf-8-sig")
        print(f"✔ Exported {len(rois_rows)} ROIs -> {prefix}_rois_summary.csv")

    # 3b. Entrapment Episodes Summary
    ent_rows = []
    for ep in episodes:
        s, e = ep["start_idx"], ep["end_idx"]
        window_ids = df.loc[s:e, "Window_ID"].tolist()
        sent_preview = None
        if has_center:
            sample_idxs = [s, (s+e)//2, e]
            sents = [str(df.loc[si, "Center_Sentence"]) for si in sample_idxs if 0 <= si < n]
            sent_preview = " || ".join(sents)
        ent_rows.append({
            "episode_id": ep["episode_id"],
            "start_idx": int(s), "end_idx": int(e),
            "start_window": ep["start_window"], "end_window": ep["end_window"],
            "duration": ep["duration"],
            "max_entr_index": ep["max_entr"], "mean_entr_index": ep["mean_entr"],
            "window_id_list": " ".join(map(str, window_ids)),
            "sample_sentences": sent_preview
        })

    if ent_rows:
        pd.DataFrame(ent_rows).to_csv(f"{prefix}_entrapment_episodes.csv", index=False, encoding="utf-8-sig")
        print(f"✔ Exported {len(ent_rows)} Entrapment Episodes -> {prefix}_entrapment_episodes.csv")

    # 3c. Flat Windows List
    flag_cols = ["Window_ID", "Narrative_Progress", val_col, aro_col, dom_col, "VAD_range", "Convergence", "Entrapment", "Entr_Index", "Episode_ID"]
    flat_df = df.loc[df["Convergence"].astype(bool) | df["Entrapment"].astype(bool), [c for c in flag_cols if c in df.columns]].copy()
    if has_center: flat_df["Center_Sentence"] = df.loc[flat_df.index, "Center_Sentence"]
    flat_df.to_csv(f"{prefix}_roi_entrapment_windows.csv", index=False, encoding="utf-8-sig")
    print(f"✔ Exported Flat Context Windows ({len(flat_df)} rows) -> {prefix}_roi_entrapment_windows.csv")

    # ============================================================
    # 4️⃣  GENERATE INDIVIDUAL PLOT
    # ============================================================
    fig, ax = plt.subplots(figsize=(14,5))
    
    # Dynamically set Y-axis limits based on scale
    ymin = min(V.min(), A.min(), D.min()) - config["y_margin"]
    ymax = max(V.max(), A.max(), D.max()) + config["y_margin"]
    ax.set_ylim(ymin, ymax)

    # Plot V/A/D Trajectories
    l1, = ax.plot(x, V, label="Valence", linewidth=1.8, color="#1f77b4")
    l2, = ax.plot(x, A, label="Arousal", linewidth=1.8, color="#ff7f0e")
    l3, = ax.plot(x, D, label="Dominance", linewidth=1.8, color="#2ca02c")

    # Define overlay patch styles
    conv_face, conv_alpha, conv_edge = "#9e9e9e", 0.22, "#4f4f4f"
    entr_face, entr_alpha, entr_edge = "#ffb3d9", 0.30, "#ff3d81"

    # Overlay Convergence ROIs (Gray)
    for (s,e) in rois:
        rect = mpatches.Rectangle((s-0.5, ymin), width=(e-s+1), height=(ymax-ymin), 
                                  facecolor=conv_face, edgecolor=conv_edge, linewidth=1.6, alpha=conv_alpha, zorder=1)
        ax.add_patch(rect)

    # Overlay Entrapment Episodes (Pink)
    for ep in episodes:
        s_pos, e_pos = ep["start_idx"], ep["end_idx"]
        rect = mpatches.Rectangle((s_pos-0.5, ymin), width=(e_pos-s_pos+1), height=(ymax-ymin), 
                                  facecolor=entr_face, edgecolor=entr_edge, linewidth=2.2, alpha=entr_alpha, zorder=5)
        ax.add_patch(rect)

    # Bring trajectory lines to the front
    l1.set_zorder(10); l2.set_zorder(10); l3.set_zorder(10)
    
    # Construct Legend
    conv_patch = mpatches.Patch(facecolor=conv_face, edgecolor=conv_edge, alpha=conv_alpha, label='VAD convergence (ROI)')
    entr_patch = mpatches.Patch(facecolor=entr_face, edgecolor=entr_edge, alpha=entr_alpha, label='Entrapment episode')
    ax.legend(handles=[l1, l2, l3, conv_patch, entr_patch], loc='upper left', framealpha=0.95)

    ax.set_xlabel("Window Index (Narrative Progress)")
    ax.set_ylabel(f"V/A/D Score ({lex_name} Scale)")
    ax.set_title(f"Affective Trajectories in The Metamorphosis ({lex_name} - {model_name})\nHighlighting Convergence ROIs & Entrapment Episodes")
    ax.grid(alpha=0.2)
    
    plt.tight_layout()
    out_plot = f"{prefix}_vad_plot.png"
    plt.savefig(out_plot, dpi=300) # High-res export for publication
    plt.close() # Free memory
    print(f"✔ Exported High-Resolution Plot -> {out_plot}")


# ============================================================
# MAIN EXECUTION ROUTINE
# ============================================================
if __name__ == "__main__":
    for lex_name, config in FILES_CONFIG.items():
        csv_path = config["path"]
        if not os.path.exists(csv_path):
            print(f"\n[SKIPPED] Source file not found: {csv_path}")
            continue
        
        print(f"\n[LOADING] Ingesting dataset: {csv_path}")
        df_main = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        # Ensure structural integrity of the dataframe
        if "Window_ID" not in df_main.columns:
            df_main["Window_ID"] = df_main.index
        if "Narrative_Progress" not in df_main.columns:
            df_main["Narrative_Progress"] = (df_main["Window_ID"] - df_main["Window_ID"].min()) / max(1, (df_main["Window_ID"].max() - df_main["Window_ID"].min()))
            
        # Process both regressors
        for model_name in MODELS:
            process_condition(df_main, lex_name, model_name, config)

    print("\n🎉 BATCH PROCESSING COMPLETED SUCCESSFULLY. All analytical artifacts have been generated.")