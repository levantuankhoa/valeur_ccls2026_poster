# Valeur

**V**AD **a**ffective tr**a**jectories for **li**terary nar**r**ative — a
computational pipeline that projects narrative prose into
Valence / Arousal / Dominance space and operationalizes the clinical
*entrapment* state (Gilbert & Allan, 1998) as a window-level
**Narrative Entrapment Index (NEI)**.

Built around sentence-transformer embeddings, lexicon-grounded Ridge
regression, and a dual-lexicon validation design (Warriner 2013 + NRC VAD
v2.1 2025). Developed for Kafka's *The Metamorphosis*; the pipeline is
corpus-agnostic.

> This repository accompanies the CCLS2026 poster **"Operationalizing
> Narrative Entrapment: Predictive Affective Trajectories via Contextual
> Sentence Embeddings in Kafka's *The Metamorphosis*."**

---

## Pipeline

```
         Lexicon (V/A/D norms)                        Text corpus
         ─────────────────────                        ────────────
              │                                             │
              │  "The word {w}."                            │ spaCy
              ▼                                             ▼
      ┌─────────────────────┐                     ┌──────────────────┐
      │        SBERT        │                     │  Sentence stream │
      │ (all-mpnet-base-v2) │                     └────────┬─────────┘
      └───────┬─────────────┘                              │
              │                                            │ 3-sent windows
              ▼                                            ▼
      ┌────────────────┐                          ┌──────────────────┐
      │   RidgeCV      │◄─ train_vad.py           │  SBERT encode    │
      │   (V / A / D)  │─► .joblib                └────────┬─────────┘
      └────────────────┘                                   │
                                                           ▼
                                                  ┌──────────────────┐
                                       encode_vad │  Ridge project   │
                                            .py   │  → V/A/D signal  │
                                                  └────────┬─────────┘
                                                           │ Savitzky–Golay
                                                           ▼
                                                  ┌──────────────────┐
                                        nei_plot  │  z-score  →      │
                                             .py  │  gated-sum NEI   │
                                                  │  → episodes + ROI│
                                                  └──────────────────┘
```

The three scripts are independent and composable:

| Script | Input | Output |
|---|---|---|
| `train_vad.py` | lexicon CSV/TSV | `.joblib` (Ridge + scalers) |
| `encode_vad.py` | text file + `.joblib` | `windows.csv` (V/A/D per window) |
| `nei_plot.py` | `windows.csv` (one or more) | NEI CSV, episodes, plots |

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/levantuankhoa/valeur_ccls2026.git
cd valeur_ccls2026
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Fetch lexicons + corpus  (see data/README.md)
#    data/warriner_2013.csv
#    data/unigrams-NRC-VAD-Lexicon-v2.1.txt
#    data/metamorphosis.txt

# 3. Run the full dual-lexicon pipeline
bash run_all.sh
```

All artefacts land in `results/`:

```
results/
├── windows_warriner.csv        # per-window V/A/D (raw + smoothed)
├── windows_nrc.csv
├── warriner.nei.csv            # per-window NEI + flags
├── warriner.episodes.csv       # entrapment episodes summary
├── warriner.rois.csv           # convergence ROIs
├── warriner.plot.png           # trajectory + overlays
├── nrc.*                       # same for NRC
└── consensus.{nei_matrix.csv,heatmap.png}
```

---

## Running individual steps

```bash
# train on one lexicon
python train_vad.py \
    --lexicon warriner \
    --path   data/warriner_2013.csv \
    --out    models/ridge_warriner.joblib

# encode text
python encode_vad.py \
    --text  data/metamorphosis.txt \
    --model models/ridge_warriner.joblib \
    --out   results/windows_warriner.csv

# NEI + plot (default: gated-sum method)
python nei_plot.py \
    --windows  results/windows_warriner.csv \
    --lexicon  warriner \
    --out-prefix results/warriner

# Ablation: compare gated-sum vs naive additive
python nei_plot.py \
    --windows results/windows_warriner.csv \
    --lexicon warriner \
    --method  add \
    --out-prefix results/warriner_addNEI

# Dual-lexicon consensus heatmap
python nei_plot.py \
    --consensus \
    --windows  results/windows_warriner.csv results/windows_nrc.csv \
    --lexicon  warriner nrc \
    --out-prefix results/consensus
```

---

## The NEI — and why gated-sum

The Narrative Entrapment Index operationalises Mehrabian–Russell's
joint-condition semantics of clinical entrapment: simultaneous
**V ↓ ∧ A ↑ ∧ D ↓**. Per window we compute rectified z-scores of valence
drop, arousal rise, and dominance drop against the text-level baseline,
then combine them.

`--method gated_sum` (default) requires every channel to clear
`--min-component` (default 0.10) before the sum is taken. This enforces
the joint condition and eliminates false positives that plague the naive
additive formulation on lexicons with near-independent V/A channels
(e.g., NRC VAD v2.1, where V↔A Pearson ≈ −0.08 empirically).

`--method add` and `--method mult` are retained for ablation.

### Cross-lexicon divergence as a validity criterion

Warriner 2013 shows moderate V↔A coupling (r ≈ −0.19 to −0.45), which
is compatible with the entrapment formula. NRC VAD 2025's near-zero
V↔A correlation is structurally incompatible with naive additive NEI —
the gate is what makes the index portable across lexicons with different
affective-space geometry. The cross-lexicon Spearman rho on the gated
NEI (printed in the consensus run) is a direct construct-validity check.

---

## Method notes

- **Sentence transformer:** `all-mpnet-base-v2` (768-d) is the default in
  `config.SBERT_MODEL` — it matches the original Feb 2025 reference run
  and the held-out probe correlations reported in the paper. A faster
  `all-MiniLM-L6-v2` (384-d) variant is supported by swapping the
  constant in `config.py`; expect a small drop in Pearson r on the
  Arousal and Dominance probes.
- **Window size:** 3 sentences, stride 1. Adjust in `config.py`.
- **Smoothing:** Savitzky–Golay, auto-sized odd window ≤ 21, polyorder 2.
- **Ridge > RF:** Ridge consistently outperforms Random Forest on the
  held-out lexicon split — empirical support for the linear-representation
  hypothesis in SBERT's geometry. RF was present in earlier drafts and
  dropped from the paper; the code path is kept minimal here.
- **Reproducibility:** `RANDOM_STATE = 42` is seeded in numpy, python's
  `random`, and torch.

### Hardware / runtime

- **Reference machine:** AMD **Ryzen 7 7800X3D** (8 cores / 16 threads,
  4.2 GHz base, 96 MB X3D cache), 32 GB DDR5 RAM, Windows 11. End-to-end
  `bash run_all.sh` (2 train + 2 encode + 3 NEI passes) completes in
  approximately 6–8 minutes on CPU at the `all-mpnet-base-v2` default.
- **GPU is optional, not required.** `utils.detect_device()` auto-selects
  CUDA via PyTorch when a compatible GPU is visible, otherwise falls back
  to CPU. Force CPU explicitly with `FORCE_CPU=1 bash run_all.sh` if you
  need deterministic behavior on a mixed machine. The hot path is the two
  SBERT encode calls (lexicon items + Kafka windows); a modern desktop
  CUDA GPU shaves these to under a minute, but the X3D cache makes CPU
  runtimes very competitive for a single-corpus pipeline.
- **No mixed-precision or batch-size tuning is required** for the default
  corpus size; `config.BATCH_SIZE = 256` works on CPU and on 8 GB+ GPUs.

---

## Citation

If you use Valeur, please cite the CCLS2026 poster (full citation after
the conference):

```bibtex
@inproceedings{khoa2026valeur,
  author    = {Lê, Văn Tuấn Khoa and Võ, Thị Phương Linh
               and Nguyễn, Thị Ngọc Trinh and Trương, Nguyễn Cát Ly},
  title     = {Operationalizing Narrative Entrapment: Predictive
               Affective Trajectories via Contextual Sentence Embeddings
               in Kafka's {\em The Metamorphosis}},
  booktitle = {Computational Literary Studies Infrastructure Conference
               (CCLS2026), Potsdam},
  year      = {2026}
}
```

---

## Authors

**Lê Văn Tuấn Khoa** (lead) — Faculty of Foreign Languages, Dalat
University, Vietnam. Independent research in computational literary
studies.

Co-authors: Võ Thị Phương Linh, Nguyễn Thị Ngọc Trinh,
Trương Nguyễn Cát Ly.

## Acknowledgments

During the preparation of this work, the author used Claude
(Anthropic) to assist with code refactoring, documentation, and
manuscript drafting. The author reviewed and edited all content and
takes full responsibility for it.

## License

MIT — see [`LICENSE`](LICENSE).
