#!/usr/bin/env bash
# Valeur — end-to-end dual-lexicon pipeline.
# Usage: bash run_all.sh
#
# Requires (place in data/):
#   warriner_2013.csv
#   unigrams-NRC-VAD-Lexicon-v2.1.txt
#   metamorphosis.txt   (or any English text corpus)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DATA="data"
MODELS="models"
RES="results"
TEXT="${DATA}/metamorphosis.txt"

mkdir -p "$MODELS" "$RES"

echo "▶ [1/6] Train Ridge (Warriner)"
python train_vad.py \
    --lexicon warriner \
    --path "${DATA}/warriner_2013.csv" \
    --out  "${MODELS}/ridge_warriner.joblib"

echo "▶ [2/6] Train Ridge (NRC)"
python train_vad.py \
    --lexicon nrc \
    --path "${DATA}/unigrams-NRC-VAD-Lexicon-v2.1.txt" \
    --out  "${MODELS}/ridge_nrc.joblib"

echo "▶ [3/6] Encode corpus via Warriner model"
python encode_vad.py \
    --text  "${TEXT}" \
    --model "${MODELS}/ridge_warriner.joblib" \
    --out   "${RES}/windows_warriner.csv"

echo "▶ [4/6] Encode corpus via NRC model"
python encode_vad.py \
    --text  "${TEXT}" \
    --model "${MODELS}/ridge_nrc.joblib" \
    --out   "${RES}/windows_nrc.csv"

echo "▶ [5/6] NEI + plot — Warriner"
python nei_plot.py \
    --windows "${RES}/windows_warriner.csv" \
    --lexicon warriner \
    --out-prefix "${RES}/warriner"

echo "▶ [6/6] NEI + plot — NRC + consensus heatmap"
python nei_plot.py \
    --windows "${RES}/windows_nrc.csv" \
    --lexicon nrc \
    --out-prefix "${RES}/nrc"

python nei_plot.py \
    --consensus \
    --windows "${RES}/windows_warriner.csv" "${RES}/windows_nrc.csv" \
    --lexicon warriner nrc \
    --out-prefix "${RES}/consensus"

echo "✔ Pipeline complete. Artefacts in ${RES}/"
