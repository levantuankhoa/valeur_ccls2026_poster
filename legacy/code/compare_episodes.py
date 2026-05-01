#!/usr/bin/env python3
"""Compare entrapment episodes and ROIs: old (Feb 2025) vs current."""
import pandas as pd
import numpy as np

def window_set(start, end):
    return set(range(start, end + 1))

def overlap_ratio(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def print_comparison(label, old_episodes, new_episodes, old_start_col, old_end_col):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    # Build window sets
    old_sets = []
    for _, row in old_episodes.iterrows():
        s, e = int(row[old_start_col]), int(row[old_end_col])
        old_sets.append((s, e, window_set(s, e)))

    new_sets = []
    for _, row in new_episodes.iterrows():
        s, e = int(row["start_window"]), int(row["end_window"])
        new_sets.append((s, e, window_set(s, e)))

    all_old = set()
    for _, _, ws in old_sets:
        all_old |= ws
    all_new = set()
    for _, _, ws in new_sets:
        all_new |= ws

    jaccard = overlap_ratio(all_old, all_new)
    print(f"\n  Old: {len(old_sets)} episodes, {len(all_old)} total windows")
    print(f"  New: {len(new_sets)} episodes, {len(all_new)} total windows")
    print(f"  Overlap (Jaccard): {jaccard:.3f}")
    print(f"  Windows in both:   {len(all_old & all_new)}")
    print(f"  Only in old:       {len(all_old - all_new)}")
    print(f"  Only in new:       {len(all_new - all_old)}")

    print(f"\n  OLD episodes:")
    for i, (s, e, _) in enumerate(old_sets):
        matched = any(len(ws & window_set(s, e)) > 0 for _, _, ws in new_sets)
        tag = "MATCHED" if matched else "MISSING"
        print(f"    #{i+1:2d}  windows {s:3d}–{e:3d} (dur={e-s+1:2d})  [{tag}]")

    print(f"\n  NEW episodes:")
    for i, (s, e, _) in enumerate(new_sets):
        matched = any(len(ws & window_set(s, e)) > 0 for _, _, ws in old_sets)
        tag = "MATCHED" if matched else "NEW"
        print(f"    #{i+1:2d}  windows {s:3d}–{e:3d} (dur={e-s+1:2d})  [{tag}]")

    # Pairwise overlap matrix
    print(f"\n  Pairwise overlap (old → new):")
    for i, (os, oe, ows) in enumerate(old_sets):
        for j, (ns, ne, nws) in enumerate(new_sets):
            iou = overlap_ratio(ows, nws)
            if iou > 0:
                print(f"    old#{i+1} ({os}-{oe}) ↔ new#{j+1} ({ns}-{ne}): IoU={iou:.3f}")


# === NRC RIDGE ===
print("\n" + "#"*70)
print("  NRC RIDGE — ENTRAPMENT EPISODES")
print("#"*70)

old_nrc = pd.read_csv("results/metamorphosis_NRC_Ridge_entrapment_episodes.csv")
new_nrc = pd.read_csv("results/new_metamorphosis_NRC_Ridge.episodes.csv")
print_comparison("NRC Ridge Entrapment", old_nrc, new_nrc, "start_window", "end_window")

# Check opening specifically
print("\n  >>> NRC — Đoạn mở đầu (windows 1–10):")
old_opening = any(int(r["start_window"]) <= 10 for _, r in old_nrc.iterrows())
new_opening = any(int(r["start_window"]) <= 10 for _, r in new_nrc.iterrows())
print(f"      Old (Feb 2025): {'CÓ detect entrapment ở mở đầu' if old_opening else 'KHÔNG detect'}")
print(f"      New (current):  {'CÓ detect entrapment ở mở đầu' if new_opening else 'KHÔNG detect'}")
if old_opening:
    for _, r in old_nrc.iterrows():
        if int(r["start_window"]) <= 10:
            print(f"      Old episode: windows {int(r['start_window'])}–{int(r['end_window'])}")
if new_opening:
    for _, r in new_nrc.iterrows():
        if int(r["start_window"]) <= 10:
            print(f"      New episode: windows {int(r['start_window'])}–{int(r['end_window'])}")


# === WARRINER RIDGE ===
print("\n" + "#"*70)
print("  WARRINER RIDGE — ENTRAPMENT EPISODES")
print("#"*70)

old_war = pd.read_csv("results/metamorphosis_Warriner_Ridge_entrapment_episodes.csv")
new_war = pd.read_csv("results/new_metamorphosis_Warriner_Ridge.episodes.csv")
print_comparison("Warriner Ridge Entrapment", old_war, new_war, "start_window", "end_window")


# === ROIs ===
print("\n" + "#"*70)
print("  NRC RIDGE — CONVERGENCE ROIs")
print("#"*70)

old_nrc_roi = pd.read_csv("results/metamorphosis_NRC_Ridge_rois_summary.csv")
new_nrc_roi = pd.read_csv("results/new_metamorphosis_NRC_Ridge.rois.csv")
# old uses ROI_id, start_window, end_window
print_comparison("NRC Ridge ROIs", old_nrc_roi, new_nrc_roi, "start_window", "end_window")

print("\n" + "#"*70)
print("  WARRINER RIDGE — CONVERGENCE ROIs")
print("#"*70)

old_war_roi = pd.read_csv("results/metamorphosis_Warriner_Ridge_rois_summary.csv")
new_war_roi = pd.read_csv("results/new_metamorphosis_Warriner_Ridge.rois.csv")
print_comparison("Warriner Ridge ROIs", old_war_roi, new_war_roi, "start_window", "end_window")
