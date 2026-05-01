# `data/` — input resources

Not committed to the repo (see `.gitignore`). Download and place the
following files here before running the pipeline.

## Lexicons

### Warriner et al. (2013) — 13,915 English words, V/A/D on 1–9 scale
- **File expected:** `warriner_2013.csv`
- **Source:** Supplementary material to
  Warriner, A. B., Kuperman, V., & Brysbaert, M. (2013). *Norms of valence,
  arousal, and dominance for 13,915 English lemmas.* Behavior Research
  Methods, 45, 1191–1207.
- **Required columns:** `Word`, `V.Mean.Sum`, `A.Mean.Sum`, `D.Mean.Sum`

### NRC VAD Lexicon v2.1 (2025) — 44,728 words, V/A/D on −1 to +1 scale
- **File expected:** `unigrams-NRC-VAD-Lexicon-v2.1.txt`
- **Source:** https://saifmohammad.com/WebPages/nrc-vad.html
- **Format:** tab-separated, no header, columns `[word, V, A, D]`

## Corpus

### Kafka, *The Metamorphosis* (default test corpus)
- **File expected:** `metamorphosis.txt`
- **Source:** Project Gutenberg (public domain English translation).
  Strip Gutenberg boilerplate from head and tail before use.
- **Substitutable:** any plain-text English corpus works. Long-form
  narrative (novel-length) gives the cleanest NEI signal.

## After download

Your `data/` directory should look like:

```
data/
├── README.md
├── warriner_2013.csv
├── unigrams-NRC-VAD-Lexicon-v2.1.txt
└── metamorphosis.txt
```

Then run `bash run_all.sh` from the repo root.
