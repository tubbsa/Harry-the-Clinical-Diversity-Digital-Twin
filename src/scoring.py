# ============================================================
# scoring.py — Bidirectional ICER-Style Equity Scoring + 0–100
# Diversity Score + Shortfall Analysis
# FINAL PRODUCTION VERSION (BUGFIX ONLY)
#
# Reference proportion update (v3):
#   Race and age groups now use U.S. Census population share
#   (Census 2025 QuickFacts) as the parity reference rather
#   than within-group CVD prevalence rates. Population share
#   was selected because prevalence and mortality data may
#   reflect systemic disparities in diagnosis and treatment
#   access, embedding existing inequities into the scoring
#   framework. Sex retains CVD mortality share (Mosca et al.
#   2011) as the reference, which captures the documented
#   gap between female CVD mortality burden and historical
#   trial underrepresentation.
#
#   Updated reference values (now in DISEASE_PREVALENCE):
#     white_pct:  0.575  (57.5% non-Hispanic White, Census 2025)
#     black_pct:  0.137  (13.7% Black alone, Census 2025)
#     asian_pct:  0.067  (6.7%  Asian alone, Census 2025)
#     aian_pct:   0.014  (1.4%  AIAN alone, Census 2025)
#     female_pct: 0.526  (CVD mortality share, Mosca et al. 2011)
#     male_pct:   0.474  (CVD mortality share, Mosca et al. 2011)
#     age65_pct:  0.180  (18.0% adults ≥65, Census 2025)
# ============================================================

import pandas as pd

from .scoring_constants import (
    DISEASE_PREVALENCE,
    RACE_GROUPS,
    SEX_GROUPS,
    AGE_GROUPS,
    DOMAIN_MAX,
)

# ============================================================
# 0–3 POINT EQUITY SCORING BASED ON DISTANCE FROM PARITY
# ============================================================

def score_pdrr(pdr):
    """
    Bidirectional equity scoring:
    Rewards near-proportional representation and penalizes
    both under- and over-representation.
    """
    if pdr is None or pdr <= 0:
        return 0

    diff = abs(pdr - 1.0)

    if diff <= 0.2:
        return 3
    elif diff <= 0.5:
        return 2
    elif diff <= 1.5:
        return 1
    else:
        return 0


# ============================================================
# MAIN ICER-STYLE 21-POINT SCORING (12 race, 6 sex, 3 age)
# ============================================================

def compute_icer_score(preds: dict, meta: dict = None, burden_override: dict = None):
    """
    Compute the equity-adjusted ICER-style diversity score.

    preds values may be floats or None.
    """

    rows = []

    # -------------------------
    # RACE DOMAIN = 12 max
    # -------------------------
    race_scores = []

    for key in RACE_GROUPS:
        trial_val = preds.get(key)
        denom = DISEASE_PREVALENCE[key]

        # ---- BUGFIX ----
        if trial_val is None or denom <= 0:
            pdr_raw = None
        else:
            pdr_raw = trial_val / denom
        # ----------------

        pdr_cap = min(pdr_raw, 1.0) if pdr_raw is not None else None
        score = score_pdrr(pdr_raw)

        race_scores.append(score)
        rows.append([key, trial_val, denom, pdr_raw, pdr_cap, score, "race"])

    # -------------------------
    # SEX DOMAIN = 6 max
    # -------------------------
    sex_scores = []

    for key in SEX_GROUPS:
        trial_val = preds.get(key)

        if burden_override is not None and key in burden_override:
            denom = burden_override[key]
        else:
            denom = DISEASE_PREVALENCE[key]

        # ---- BUGFIX ----
        if trial_val is None or denom <= 0:
            pdr_raw = None
        else:
            pdr_raw = trial_val / denom
        # ----------------

        pdr_cap = min(pdr_raw, 1.0) if pdr_raw is not None else None
        score = score_pdrr(pdr_raw)

        sex_scores.append(score)
        rows.append([key, trial_val, denom, pdr_raw, pdr_cap, score, "sex"])

    # -------------------------
    # AGE DOMAIN = 3 max
    # -------------------------
    age_scores = []

    for key in AGE_GROUPS:
        trial_val = preds.get(key)
        denom = DISEASE_PREVALENCE[key]

        # ---- BUGFIX ----
        if trial_val is None or denom <= 0:
            pdr_raw = None
        else:
            pdr_raw = trial_val / denom
        # ----------------

        pdr_cap = min(pdr_raw, 1.0) if pdr_raw is not None else None
        score = score_pdrr(pdr_raw)

        age_scores.append(score)
        rows.append([key, trial_val, denom, pdr_raw, pdr_cap, score, "age"])

    # -------------------------
    # DOMAIN TOTALS
    # -------------------------
    race_total = min(sum(race_scores), DOMAIN_MAX["race"])
    sex_total  = min(sum(sex_scores),  DOMAIN_MAX["sex"])
    age_total  = min(sum(age_scores),  DOMAIN_MAX["age"])

    total = race_total + sex_total + age_total

    # -------------------------
    # BREAKDOWN TABLE
    # -------------------------
    breakdown = pd.DataFrame(
        rows,
        columns=[
            "component",
            "value",
            "population_reference",
            "PDRR_raw",
            "PDRR",
            "score",
            "domain",
        ],
    )

    return float(total), breakdown


# ============================================================
# UNIFIED 0–100 DIVERSITY SCORE (for GUI + Twin Optimization)
# ============================================================

def compute_diversity_score(preds: dict):
    """
    Converts ICER (0–21) to a 0–100 diversity score.
    """

    icer_total, breakdown = compute_icer_score(preds)

    diversity_score = (icer_total / 21.0) * 100.0

    shortfall_rows = []

    for key in DISEASE_PREVALENCE:
        trial_val = preds.get(key)
        denom = DISEASE_PREVALENCE[key]

        # ---- BUGFIX ----
        if trial_val is None:
            shortfall = None
        else:
            shortfall = trial_val - denom
        # ----------------

        shortfall_rows.append([key, trial_val, denom, shortfall])

    shortfalls = pd.DataFrame(
        shortfall_rows,
        columns=["component", "predicted", "population_reference", "shortfall"],
    )

    return {
        "icer_total": float(icer_total),
        "icer_breakdown": breakdown,
        "diversity_score": float(diversity_score),
        "shortfalls": shortfalls,
    }
