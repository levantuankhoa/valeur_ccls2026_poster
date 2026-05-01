#!/usr/bin/env python3
"""Detail all entrapment episodes with full sentences, compare Warriner vs NRC."""
import pandas as pd
import numpy as np

def load_nei(windows_csv, nei_csv):
    win = pd.read_csv(windows_csv, encoding="utf-8-sig")
    nei = pd.read_csv(nei_csv, encoding="utf-8-sig")
    return nei

def print_episodes(nei_df, label):
    episodes = []
    if "Episode_ID" not in nei_df.columns:
        return episodes
    for eid in sorted(nei_df["Episode_ID"].unique()):
        if eid == 0:
            continue
        ep = nei_df[nei_df["Episode_ID"] == eid].copy()
        episodes.append({
            "id": eid,
            "start": int(ep["Window_ID"].min()),
            "end": int(ep["Window_ID"].max()),
            "duration": len(ep),
            "max_nei": ep["NEI"].max(),
            "mean_nei": ep["NEI"].mean(),
            "windows": set(range(int(ep["Window_ID"].min()), int(ep["Window_ID"].max()) + 1)),
            "rows": ep,
        })

    print(f"\n{'#'*80}")
    print(f"  {label} — {len(episodes)} entrapment episodes")
    print(f"{'#'*80}")

    for ep in episodes:
        print(f"\n  Episode #{ep['id']}  |  Windows {ep['start']}–{ep['end']}  |  Duration: {ep['duration']}  |  Max NEI: {ep['max_nei']:.3f}")
        print(f"  {'─'*74}")
        for _, row in ep["rows"].iterrows():
            nei_val = row["NEI"]
            wid = int(row["Window_ID"])
            sent = row["Center_Sentence"]
            if len(str(sent)) > 120:
                sent = str(sent)[:117] + "..."
            print(f"    W{wid:3d}  NEI={nei_val:5.2f}  │ {sent}")

    return episodes


# Load
war_nei = pd.read_csv("results/new_metamorphosis_Warriner_Ridge.nei.csv", encoding="utf-8-sig")
nrc_nei = pd.read_csv("results/new_metamorphosis_NRC_Ridge.nei.csv", encoding="utf-8-sig")

war_eps = print_episodes(war_nei, "WARRINER Ridge (gated_sum)")
nrc_eps = print_episodes(nrc_nei, "NRC Ridge (gated_sum)")

# Overlap analysis
print(f"\n{'#'*80}")
print(f"  OVERLAP ANALYSIS: Warriner vs NRC")
print(f"{'#'*80}")

war_all = set()
for ep in war_eps:
    war_all |= ep["windows"]
nrc_all = set()
for ep in nrc_eps:
    nrc_all |= ep["windows"]

both = war_all & nrc_all
only_war = war_all - nrc_all
only_nrc = nrc_all - war_all
jaccard = len(both) / len(war_all | nrc_all) if (war_all | nrc_all) else 0

print(f"\n  Warriner: {len(war_all)} windows in {len(war_eps)} episodes")
print(f"  NRC:      {len(nrc_all)} windows in {len(nrc_eps)} episodes")
print(f"  Both:     {len(both)} windows")
print(f"  Only Warriner: {len(only_war)} windows")
print(f"  Only NRC:      {len(only_nrc)} windows")
print(f"  Jaccard:  {jaccard:.3f}")

print(f"\n  Consensus windows (both lexicons):")
if both:
    for w in sorted(both):
        war_row = war_nei[war_nei["Window_ID"] == w].iloc[0]
        nrc_row = nrc_nei[nrc_nei["Window_ID"] == w].iloc[0]
        sent = str(war_row["Center_Sentence"])
        if len(sent) > 100:
            sent = sent[:97] + "..."
        print(f"    W{w:3d}  WAR_NEI={war_row['NEI']:5.2f}  NRC_NEI={nrc_row['NEI']:5.2f}  │ {sent}")
else:
    print("    (none)")

# Apple-throwing scene search
print(f"\n{'#'*80}")
print(f"  TÌM KHÚC NÉM TÁO (apple scene)")
print(f"{'#'*80}")

for name, df in [("Warriner", war_nei), ("NRC", nrc_nei)]:
    apple_mask = df["Center_Sentence"].str.contains("apple", case=False, na=False) | \
                 df["Full_Window_Text"].str.contains("apple", case=False, na=False)
    apple_rows = df[apple_mask]
    print(f"\n  {name} — windows mentioning 'apple': {len(apple_rows)}")
    for _, row in apple_rows.iterrows():
        wid = int(row["Window_ID"])
        nei = row["NEI"]
        entr = int(row["Entrapment"])
        ep = int(row["Episode_ID"])
        sent = str(row["Center_Sentence"])
        if len(sent) > 100:
            sent = sent[:97] + "..."
        tag = f"ENTRAPMENT ep#{ep}" if entr else "not flagged"
        print(f"    W{wid:3d}  NEI={nei:5.2f}  [{tag}]  │ {sent}")
