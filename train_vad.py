#!/usr/bin/env python3
"""
Valeur — Step 1: train a Ridge regressor to project SBERT embeddings
into Valence / Arousal / Dominance space.

Input:   a VAD lexicon (warriner or nrc format)
Output:  a .joblib artefact containing the fitted Ridge models + scalers,
         plus stdout diagnostics (Pearson r, Spearman rho, RMSE, MAE,
         bootstrap 95% CI, permutation p).

Example:
    python train_vad.py --lexicon warriner --path data/warriner_2013.csv \
        --out models/ridge_warriner.joblib
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config
from utils import (
    encode_texts,
    load_lexicon,
    load_sbert,
    set_global_seed,
)


# -----------------------------------------------------------------------------
# Statistical diagnostics
# -----------------------------------------------------------------------------
def bootstrap_r_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int, alpha: float = 0.05):
    """Percentile bootstrap 95% CI for Pearson r."""
    n = len(y_true)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        try:
            stats[b] = pearsonr(y_true[idx], y_pred[idx])[0]
        except Exception:
            stats[b] = 0.0
    low  = float(np.percentile(stats, 100 * alpha / 2))
    high = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return float(np.mean(stats)), (low, high)


def permutation_test_r(y_true: np.ndarray, y_pred: np.ndarray, n_perm: int) -> float:
    """Two-sided permutation p-value for Pearson r."""
    orig = pearsonr(y_true, y_pred)[0]
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(y_pred)
        try:
            r_perm = pearsonr(y_true, perm)[0]
        except Exception:
            r_perm = 0.0
        if abs(r_perm) >= abs(orig):
            count += 1
    return (count + 1) / (n_perm + 1)


# -----------------------------------------------------------------------------
# Core training routine
# -----------------------------------------------------------------------------
def train(
    lexicon_name: str,
    lexicon_path: Path,
    out_path: Path,
    n_boot: int = config.N_BOOTSTRAP,
    n_perm: int = config.N_PERMUTATION,
) -> dict:
    t0 = time.time()
    set_global_seed()

    # 1. Load lexicon and encode words
    lex_df = load_lexicon(lexicon_name, lexicon_path)
    words = lex_df["word"].tolist()
    y = lex_df[["V", "A", "D"]].values.astype(float)

    sbert = load_sbert()
    templates = [config.TEMPLATE.format(w) for w in words]
    print(f"[encode] {len(templates):,} lexicon items → SBERT")
    X = encode_texts(sbert, templates)

    # 2. Split, scale
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    x_scaler = StandardScaler().fit(X_tr)
    y_scaler = StandardScaler().fit(y_tr)
    X_tr_s = x_scaler.transform(X_tr)
    X_te_s = x_scaler.transform(X_te)
    y_tr_s = y_scaler.transform(y_tr)

    # 3. Fit one RidgeCV per affective dimension
    print(f"[ridge] fitting 3 × RidgeCV (alphas={len(config.ALPHAS)} candidates, cv=5)")
    models = []
    for dim in range(3):
        rc = RidgeCV(alphas=list(config.ALPHAS), cv=5)
        rc.fit(X_tr_s, y_tr_s[:, dim])
        models.append(rc)

    # 4. Evaluate on held-out set (inverse-transform back to lexicon scale)
    preds_s = np.column_stack([m.predict(X_te_s) for m in models])
    preds   = y_scaler.inverse_transform(preds_s)

    diagnostics = {}
    print(f"\n[eval] held-out performance ({lexicon_name})")
    for i, label in enumerate(["Valence", "Arousal", "Dominance"]):
        yt = y_te[:, i]
        yp = preds[:, i]
        r  = float(pearsonr(yt, yp)[0])
        rho = float(spearmanr(yt, yp).correlation)
        rmse = math.sqrt(mean_squared_error(yt, yp))
        mae  = mean_absolute_error(yt, yp)
        r_mean, (lo, hi) = bootstrap_r_ci(yt, yp, n_boot=n_boot)
        p = permutation_test_r(yt, yp, n_perm=n_perm)
        diagnostics[label] = {
            "pearson_r":  r,
            "spearman_rho": rho,
            "rmse": rmse,
            "mae":  mae,
            "bootstrap_r_mean": r_mean,
            "bootstrap_ci_95":  [lo, hi],
            "permutation_p":    p,
            "chosen_alpha":     float(models[i].alpha_),
        }
        print(
            f"  {label:9s} r={r:.3f}  rho={rho:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}"
            f"  | 95% CI [{lo:.3f}, {hi:.3f}]  p={p:.4f}  alpha={models[i].alpha_:.2g}"
        )

    # 5. Persist artefact
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artefact = {
        "lexicon_name": lexicon_name,
        "sbert_model":  config.SBERT_MODEL,
        "template":     config.TEMPLATE,
        "models":       models,        # list[RidgeCV] for V, A, D
        "x_scaler":     x_scaler,
        "y_scaler":     y_scaler,
        "diagnostics":  diagnostics,
        "n_train":      int(len(X_tr)),
        "n_test":       int(len(X_te)),
        "scale":        config.LEXICONS[lexicon_name]["scale"],
    }
    joblib.dump(artefact, out_path)
    print(f"\n[save] {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[done] elapsed {time.time() - t0:.1f}s")
    return diagnostics


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train Ridge VAD regressor (Valeur step 1/3)")
    ap.add_argument("--lexicon", required=True, choices=list(config.LEXICONS.keys()),
                    help="lexicon format: 'warriner' (1-9) or 'nrc' (-1 to +1)")
    ap.add_argument("--path", required=True, type=Path, help="path to lexicon file")
    ap.add_argument("--out",  required=True, type=Path, help="output .joblib path")
    ap.add_argument("--bootstrap", type=int, default=config.N_BOOTSTRAP)
    ap.add_argument("--perm",      type=int, default=config.N_PERMUTATION)
    args = ap.parse_args()

    diagnostics = train(
        lexicon_name=args.lexicon,
        lexicon_path=args.path,
        out_path=args.out,
        n_boot=args.bootstrap,
        n_perm=args.perm,
    )
    # Also dump diagnostics next to the joblib as JSON for easy inspection
    json_path = args.out.with_suffix(".diagnostics.json")
    with open(json_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"[save] {json_path}")


if __name__ == "__main__":
    main()
