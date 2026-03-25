# ============================================================
# src/predictor.py
# ============================================================
# Fixes applied (2026-03-25):
#
#   1. build_text_embedding: corrected payload key names to match
#      payload_builder.py output:
#        "conditions_text"        → "Conditions"
#        "interventions_text"     → "Interventions"
#        "primary_outcome_text"   → "Primary Outcome Measures"
#        "secondary_outcome_text" → "Secondary Outcome Measures"
#      Previously these four fields silently fell back to ""
#      on every call, meaning the embedding was built from
#      inclusion and exclusion text only.
#
#   2. predict_proportions: CAT_COLS key lookup now remaps
#      payload_builder.py keys to the lowercase/underscore
#      keys expected by CAT_COLS and the frozen encoder:
#        "Sponsor"      → "sponsor"
#        "Collaborators"→ "collaborators"
#        "Phases"       → "phases"
#        "Funder Type"  → "funder_type"
#      (allocation, intervention_model, masking, primary_purpose,
#       eligibility_sex, study_type already matched.)
#      Previously mismatched keys silently defaulted to "Unknown",
#      meaning sponsor, collaborators, phases, and funder type
#      had no effect on model predictions.
#
#   3. Removed duplicate dead code block (second hurdle loop and
#      second return statement) that was unreachable.
# ============================================================

import os
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
from catboost import CatBoostRegressor
from sentence_transformers import SentenceTransformer

try:
    from .schema import SCHEMA_VERSION, coerce_demo_keys
except ImportError:  # pragma: no cover
    from schema import SCHEMA_VERSION, coerce_demo_keys


# src/predictor.py → repo root = parents[1]
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model"

MODEL_PATH         = MODEL_DIR / "cb_model_multituned.cbm"
ENCODER_PATH       = MODEL_DIR / "encoder.pkl"
CAT_COLS_PATH      = MODEL_DIR / "CAT_COLS.pkl"
NUM_COLS_PATH      = MODEL_DIR / "NUM_COLS.pkl"
TARGET_COLS_PATH   = MODEL_DIR / "TARGET_COLS.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "FEATURE_NAMES.pkl"

HURDLE_CLF_PATH    = MODEL_DIR / "hurdle_clf.pkl"
HURDLE_REG_PATH    = MODEL_DIR / "hurdle_reg.pkl"


# ------------------------------------------------------------
# LOAD MODELS & ARTIFACTS (ON IMPORT)
# ------------------------------------------------------------
def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required artifact '{label}' at: {path}. "
            "Ensure model artifacts are committed to the repository."
        )
    if path.is_file() and path.stat().st_size < 16:
        raise RuntimeError(
            f"Artifact '{label}' looks too small/corrupt: {path} ({path.stat().st_size} bytes)"
        )


print(f"[predictor] CWD={os.getcwd()}")
print(f"[predictor] REPO_ROOT={REPO_ROOT}")
print(f"[predictor] MODEL_DIR={MODEL_DIR}")

_require_file(MODEL_PATH, "cb_model_multituned.cbm")
print(f"[predictor] MODEL_PATH={MODEL_PATH} (bytes={MODEL_PATH.stat().st_size})")

model = CatBoostRegressor()
model.load_model(str(MODEL_PATH))

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

with open(CAT_COLS_PATH, "rb") as f:
    CAT_COLS = pickle.load(f)

with open(NUM_COLS_PATH, "rb") as f:
    NUM_COLS = pickle.load(f)

with open(TARGET_COLS_PATH, "rb") as f:
    TARGET_COLS = pickle.load(f)

with open(FEATURE_NAMES_PATH, "rb") as f:
    FEATURE_NAMES = pickle.load(f)

with open(HURDLE_CLF_PATH, "rb") as f:
    hurdle_clf = pickle.load(f)

with open(HURDLE_REG_PATH, "rb") as f:
    hurdle_reg = pickle.load(f)


# ------------------------------------------------------------
# LOAD OOD ARTIFACTS
# ------------------------------------------------------------
OOD_MEAN = np.load(os.path.join(MODEL_DIR, "ood_mean.npy"))
OOD_STD  = np.load(os.path.join(MODEL_DIR, "ood_std.npy"))
OOD_COLS = np.load(
    os.path.join(MODEL_DIR, "ood_cols.npy"),
    allow_pickle=True
).tolist()

print("DEBUG predictor OOD_MEAN shape:", OOD_MEAN.shape)
print("DEBUG predictor OOD_STD shape:", OOD_STD.shape)
print("DEBUG predictor OOD_COLS:", OOD_COLS)


# ------------------------------------------------------------
# TEXT EMBEDDER
# ------------------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ------------------------------------------------------------
# Key mapping: payload_builder.py keys → CAT_COLS keys
#
# payload_builder.py uses mixed-case and spaced keys for
# categoricals. CAT_COLS uses lowercase_underscore. This map
# is used at inference time to look up payload values by their
# payload_builder key and assign them to the correct CAT_COL.
# Fields not listed here are looked up by col name directly
# (pass-through, already matching).
# ------------------------------------------------------------
_PAYLOAD_TO_CAT = {
    "eligibility_sex":    "eligibility_sex",
    "Sponsor":            "sponsor",
    "Collaborators":      "collaborators",
    "Phases":             "phases",
    "Funder Type":        "funder_type",
    "Study Type":         "study_type",
    "allocation":         "allocation",
    "intervention_model": "intervention_model",
    "masking":            "masking",
    "primary_purpose":    "primary_purpose",
}

# Reverse map: CAT_COL name → payload key
_CAT_TO_PAYLOAD = {v: k for k, v in _PAYLOAD_TO_CAT.items()}


# ============================================================
# HELPERS
# ============================================================

def build_text_embedding(payload: Dict[str, Any]) -> np.ndarray:
    """
    Build 384-dim MiniLM embedding from trial text fields.

    Key names match payload_builder.py output exactly:
      inclusion_text           — joined inclusion criteria
      exclusion_text           — joined exclusion criteria
      Conditions               — joined conditions
      Interventions            — joined interventions
      Primary Outcome Measures — joined primary outcomes
      Secondary Outcome Measures — joined secondary outcomes
    """
    text_fields = [
        payload.get("inclusion_text", ""),
        payload.get("exclusion_text", ""),
        payload.get("Conditions", ""),
        payload.get("Interventions", ""),
        payload.get("Primary Outcome Measures", ""),
        payload.get("Secondary Outcome Measures", ""),
    ]
    joined = " ".join(t for t in text_fields if isinstance(t, str))
    return embedder.encode(joined).reshape(1, -1)


def check_ood(x_vec: np.ndarray):
    """
    Restricted diagonal OOD check using stable metadata features only.
    x_vec must already be in final feature space order.
    """
    col_idx = [FEATURE_NAMES.index(c) for c in OOD_COLS]
    x_restricted = x_vec[:, col_idx]
    z_vec = np.abs((x_restricted - OOD_MEAN) / OOD_STD)
    z_max = float(np.max(z_vec))
    is_ood = bool(z_max > 3.5)
    return is_ood, z_max


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

def predict_proportions(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict demographic proportions for a proposed clinical trial.

    Parameters
    ----------
    payload : dict
        Output of payload_builder.build_payload(). Must contain
        text fields, categorical fields, and numeric fields as
        defined in payload_builder.py.

    Returns
    -------
    dict with keys:
        unreliable_projection : bool
        ood_score             : float
        preds                 : dict of demographic proportions
        _schema               : str
    """

    # -----------------------
    # 1. TEXT → EMBEDDING
    # -----------------------
    X_text = build_text_embedding(payload)

    # -----------------------
    # 2. CATEGORICAL FEATURES
    # -----------------------
    # For each col in CAT_COLS, look up its value from payload
    # using the payload_builder key (via _CAT_TO_PAYLOAD).
    # Falls back to col name directly if not in the map,
    # then to "Unknown" if not found in payload at all.
    cat_vals = []
    for col in CAT_COLS:
        payload_key = _CAT_TO_PAYLOAD.get(col, col)
        cat_vals.append(payload.get(payload_key, "Unknown"))
    X_cat = encoder.transform([cat_vals])

    # -----------------------
    # 3. NUMERIC FEATURES
    # -----------------------
    # NUM_COLS keys match payload_builder.py exactly:
    #   eligibility_min_age_yrs, eligibility_max_age_yrs,
    #   min_age_missing, max_age_missing,
    #   n_sites, n_us_states, n_us_regions
    X_num = np.array([[payload.get(c, 0) for c in NUM_COLS]])

    # -----------------------
    # 4. FINAL FEATURE VECTOR
    # -----------------------
    X = np.hstack([X_text, X_cat, X_num])

    # -----------------------
    # 5. OOD CHECK
    # -----------------------
    is_ood, z_score = check_ood(X)

    # -----------------------
    # 6. BASE MODEL PREDICTION
    # -----------------------
    raw_preds = model.predict(X).flatten()
    preds = dict(zip(TARGET_COLS, raw_preds))

    # -----------------------
    # 7. HURDLE MODELS (RARE TARGETS)
    # -----------------------
    for label, clf in hurdle_clf.items():
        present = clf.predict(X)[0]
        if present == 0:
            preds[label] = 0.0
        else:
            preds[label] = float(hurdle_reg[label].predict(X)[0])

    # -----------------------
    # 8. CANONICALIZE KEYS
    # -----------------------
    preds_frac = coerce_demo_keys(preds)
    for k in ["aian_pct", "age65_pct"]:
        preds_frac.setdefault(k, preds.get(k, 0.0))

    return {
        "unreliable_projection": bool(is_ood),
        "ood_score": float(z_score),
        "preds": preds_frac,
        "_schema": SCHEMA_VERSION,
    }
