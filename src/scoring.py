# ============================================================
# scoring.py — Bidirectional ICER-Style Equity Scoring + 0–100
# Diversity Score + Shortfall Analysis
# FINAL PRODUCTION VERSION (BUGFIX ONLY)
#
# Sex domain reference change (v2):
#   Previously used DISEASE_PREVALENCE for sex (female: 5.8%,
#   male: 7.8%). Because these are condition prevalence rates
#   rather than population sex proportions, any realistic
#   enrollment split produced PDRRs far above 1.0, placing all
#   configurations outside the scoring bands and yielding a
#   sex domain score of 0/6 in every case.
#
#   The sex domain now uses SEX_BURDEN_MORTALITY as the default
#   reference (female: 52.6%, male: 47.4%), derived from the
#   share of U.S. cardiovascular deaths by sex (Mosca et al.,
#   Circulation 2011). This reference produces PDRRs near 1.0
#   for typical enrollment splits, making the sex score
#   sensitive to actual trial design decisions.
#
#   The burden_override parameter is retained for
#   backward compatibility and can still be used to pass
#   an alternative reference if needed.
# ============================================================

import pandas as pd

from .scoring_constants import (
    DISEASE_PREVALENCE,
    SEX_BURDEN_MORTALITY,
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
    # Default reference: SEX_BURDEN_MORTALITY (cardiovascular
    # mortality share by sex, Mosca et al. 2011).
    # burden_override can supply an alternative reference if needed.
    sex_scores = []

    for key in SEX_GROUPS:
        trial_val = preds.get(key)

        if burden_override is not None and key in burden_override:
            denom = burden_override[key]
        else:
            denom = SEX_BURDEN_MORTALITY[key]

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
            "disease_prevalence",
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
        columns=["component", "predicted", "disease_prevalence", "shortfall"],
    )

    return {
        "icer_total": float(icer_total),
        "icer_breakdown": breakdown,
        "diversity_score": float(diversity_score),
        "shortfalls": shortfalls,
    }

