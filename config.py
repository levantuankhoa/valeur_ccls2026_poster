"""
Valeur — shared configuration constants.

All pipeline defaults live here. Override via CLI flags in the individual
scripts; do not edit paths at runtime.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT_DIR    = Path(__file__).resolve().parent
DATA_DIR    = ROOT_DIR / "data"
MODELS_DIR  = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# -----------------------------------------------------------------------------
# SBERT / embedding
# -----------------------------------------------------------------------------
SBERT_MODEL    = "all-mpnet-base-v2"   # 768-d; matches original Feb 2025 pipeline
TEMPLATE       = "The word {}."        # encoding convention for lexicon items
BATCH_SIZE     = 256                   # reduce if OOM on small GPUs
RANDOM_STATE   = 42

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
TEST_SIZE         = 0.20
ALPHAS            = (1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6)  # RidgeCV grid
N_BOOTSTRAP       = 2000
N_PERMUTATION     = 2000
MIN_WORD_LENGTH   = 1   # drop single-char artefacts if desired

# -----------------------------------------------------------------------------
# Sliding-window encoding
# -----------------------------------------------------------------------------
WINDOW_SIZE          = 3    # sentences per window
STEP                 = 1    # stride
MIN_SENTENCE_LENGTH  = 15   # skip sub-15-char sentences (likely fragments)
SAVGOL_WINDOW_MAX    = 21   # Savitzky-Golay smoothing window (auto-odd, auto-clamp)
SAVGOL_POLYORDER     = 2

# -----------------------------------------------------------------------------
# NEI — Narrative Entrapment Index
# -----------------------------------------------------------------------------
#   method="gated_sum" is the corrected default: requires all three z-components
#   (zV_drop, zA_rise, zD_drop) to exceed MIN_COMPONENT before summing. This
#   eliminates the ~74% false-positive rate of the naive "add" method on
#   lexicons with near-independent V/A channels (e.g., NRC VAD v2.1).
NEI_METHOD        = "gated_sum"   # {"gated_sum", "add", "mult"}
NEI_MIN_COMPONENT = 0.10          # z-threshold per channel for gated_sum
NEI_PERCENTILE    = 0.95          # top 5% of windows flagged as entrapment
BASELINE_METHOD   = "median"      # {"median", "mean"}
MIN_EPISODE_DURATION = 2          # contiguous windows required for an episode

# -----------------------------------------------------------------------------
# Convergence ROI (VAD range threshold — scale-dependent)
# -----------------------------------------------------------------------------
CONVERGENCE_THRESHOLD = {
    "warriner": 0.50,   # scale 1–9
    "nrc":      0.15,   # scale −1 to +1
}

# -----------------------------------------------------------------------------
# Lexicon registry — declarative; add new entries here
# -----------------------------------------------------------------------------
LEXICONS = {
    "warriner": {
        "format":    "csv",
        "word_col":  "Word",
        "v_col":     "V.Mean.Sum",
        "a_col":     "A.Mean.Sum",
        "d_col":     "D.Mean.Sum",
        "scale":     "1-9",
        "reference": "Warriner, Kuperman & Brysbaert (2013)",
    },
    "nrc": {
        "format":    "tsv",
        "word_col":  0,   # positional: NRC file has no header
        "v_col":     1,
        "a_col":     2,
        "d_col":     3,
        "scale":     "-1 to +1",
        "reference": "Mohammad (2025), NRC VAD Lexicon v2.1",
    },
}
