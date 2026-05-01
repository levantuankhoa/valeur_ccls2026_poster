# `legacy/` — Feb 2025 artefacts (do not use)

These files predate the April 2026 refactor. They are kept for **historical / audit purposes only** — to demonstrate the evolution path from the original "Noema / Project Kafka" draft to the current Valeur pipeline.

**Do not run any of this code on current outputs.** Schema mismatches and the rejected `ENTR_METHOD = "add"` formula will produce results that contradict the paper.

For the full evolution narrative — schema diffs, NEI formula change, project rename, RF removal — see `../LEGACY_CONTEXT.md` at the repo root.

---

## Contents

### `code/`

Seven Python scripts from the Feb 2025 monolithic pipeline + the regression-test scripts that compared old vs. new outputs during the April 2026 refactor.

| File | What it was for | Why deprecated |
|---|---|---|
| `vad_roi_plot.py` | Single-condition ROI + entrapment processor | Uses `ENTR_METHOD = "add"` (rejected formula); reads old `metamorphosis_vad_*_windows.csv` filename schema |
| `vad_roi_plot_batch.py` | Batch processor for {Warriner, NRC} × {Ridge, RF} | Same as above, plus references RF column suffixes that no longer exist |
| `compare_results.py` | Regression test: current Ridge VAD trajectories vs. Feb 2025 outputs | Reads both old (`Valence_Ridge_Smooth`) and new (`Valence_Smooth`) column schemas; useful only during transition |
| `compare_episodes.py` | Regression test: episode/ROI Jaccard old vs. new | Same |
| `diagnose_method.py` | Reverse-engineered which NEI method Feb 2025 used | Question answered (it was `add`); script is single-use |
| `detail_episodes.py` | Pretty-printed episode details with sentence context | Reads the transitional `new_metamorphosis_*` filename prefix that has since been removed |
| `plot_convergence.py` | Plotted VAD trajectories with convergence ROI overlays | Same prefix issue; functionality now in `nei_plot.py:run_single` |

### `results_feb2025/`

25 result artefacts produced by the Feb 2025 monolithic pipeline:

- `metamorphosis_<Lex>_RF_*.{csv,png}` — Random Forest probe outputs (RF removed from production in April 2026)
- `metamorphosis_<Lex>_Ridge_*.{csv,png}` — Ridge probe outputs under the old filename schema and `ENTR_METHOD = "add"`
- `comparison_old_vs_new_ridge.png` / `scatter_old_vs_new_ridge.png` / `diff_distribution_old_vs_new_ridge.png` — produced by `compare_results.py` during the refactor; document the Ridge trajectory delta between Feb 2025 and April 2026
- `nei_consensus_overlay.png` / `vad_convergence_both_lexicons.png` / `vad_range_convergence.png` — exploratory plots from the refactor period

---

## Why keep them

1. **Reviewer accountability.** If a reviewer asks "what changed between drafts?", the diff is here.
2. **Reproducibility of the evolution claim.** Section 2.2 of `data/NEI_Pipeline_Reference.md` argues `ENTR_METHOD = "add"` was rejected because of a Window 1 false positive on Gregor's awakening. The `metamorphosis_*_Ridge_entrapment_episodes.csv` files in `results_feb2025/` are the empirical evidence for that claim.
3. **Citation trail for the cross-lexicon match.** The `comparison_old_vs_new_ridge.png` shows the trajectories are nearly identical between Feb 2025 and April 2026, which validates that the refactor did not silently change the underlying VAD signal.

---

## When to delete this folder

Once the CCLS 2026 poster is presented and the JCLS journal version is camera-ready, this folder can be safely deleted — the canonical reference is `data/NEI_Pipeline_Reference.md` and `LEGACY_CONTEXT.md`. Until then, leave it in place.
