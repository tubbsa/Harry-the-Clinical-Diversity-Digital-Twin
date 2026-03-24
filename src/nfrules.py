# ============================================================
# nfrules.py — Neuro-Fuzzy Policy Layer (SUBGROUP-AWARE)
# Abigail Tubbs
#
# Purpose:
#   Translate model-predicted CDR score + subgroup proportions into
#   interpretable, actionable trial design recommendations.
#
# Design:
#   • CatBoost     = learning
#   • ICER/CDR     = scoring
#   • Fuzzy system = policy interpretation (NOT trained)
#
# Gap metric:
#   All gaps are expressed as PDRR = trial_val / population_reference,
#   matching the scoring logic in scoring.py / score_pdrr().
#   PDRR = 1.0 means exact parity with the population reference.
#   PDRR < 1.0 means under-represented relative to the reference.
#   Recommendations trigger when PDRR falls below a group threshold.
#
# Reference proportions (from scoring_constants.py DISEASE_PREVALENCE):
#   Race and age groups use U.S. Census population share (2025).
#   Sex groups use CVD mortality share (Mosca et al., Circulation 2011).
#
#   Group        Reference   Source
#   --------------------------------------------------------
#   White        0.575       Census 2025 (non-Hispanic White alone)
#   Black        0.137       Census 2025
#   Asian        0.067       Census 2025
#   AIAN         0.014       Census 2025
#   Female       0.526       CVD mortality share (Mosca et al.)
#   Male         0.474       CVD mortality share (Mosca et al.)
#   Age65+       0.180       Census 2025
#
# Threshold rationale — grounded in observed PDRR distributions
# across 1,290 U.S. cardiovascular trials (training dataset),
# computed against Census population share / CVD mortality share
# references:
#
#   Group        Typical PDRR range   Threshold   Rationale
#   --------------------------------------------------------
#   White        1.2 -- 1.5           0.80        Over-represented vs Census
#                                                  in most trials; threshold
#                                                  fires only on unusual
#                                                  under-enrollment
#   Male         1.0 -- 1.2           0.80        Near or above parity;
#                                                  same rationale as White
#   Black        0.4 -- 1.0           0.90        Chronically under-
#                                                  represented; threshold
#                                                  fires earlier to catch
#                                                  gaps sooner
#   Female       0.8 -- 1.0           0.90        Near parity but
#                                                  historically under-
#                                                  represented; matches
#                                                  Black sensitivity
#   Asian        0.0 -- 0.5           0.95        Structural absence in
#                                                  majority of trials;
#                                                  most sensitive threshold
#   AIAN         0.0 -- 0.1           0.95        Near-zero enrollment in
#                                                  nearly all trials;
#                                                  most sensitive threshold
#   Age65+       0.1 -- 0.8           0.95        Structurally under-
#                                                  enrolled relative to
#                                                  Census share in most
#                                                  trials; most sensitive
#
#   Principle: the worse the historical representation relative to
#   Census population share, the higher the threshold and the more
#   sensitive the recommendation trigger. This grounds differential
#   sensitivity in observed evidence rather than arbitrary design.
#
#   NOTE: Median PDRR values previously reported in this file were
#   computed against CVD prevalence references and are no longer
#   valid under Census population share references. Observed median
#   PDRRs should be recomputed from training data against the current
#   DISEASE_PREVALENCE constants before updating _OBSERVED_MEDIAN_PDRR.
#
# SHAP grounding (companion study, Figure 3A/B/C):
#
#   Per-TARGET feature importance (Figure 3A heatmap):
#     white_pct   — protocol text > recruitment scope > eligibility
#     black_pct   — protocol text > recruitment scope > eligibility
#     asian_pct   — eligibility constraints relatively more important;
#                   hurdle model (presence-driven)
#     aian_pct    — eligibility constraints dominant; hurdle model
#     female_pct  — eligibility sex features overwhelmingly dominant
#     male_pct    — eligibility sex features overwhelmingly dominant
#     age65_pct   — age bounds + recruitment scope both important
#
#   Top individual structured features (Figure 3B):
#     1.  eligibility_sex_FEMALE
#     2.  n_us_regions
#     3.  n_us_states
#     4.  eligibility_min_age_yrs
#     5.  eligibility_sex_ALL
#     6.  eligibility_max_age_yrs
#     7.  eligibility_sex_MALE
#     8.  n_sites
#     9.  study_type_Unknown
#     10. max_age_missing / min_age_missing
#
#   Per-feature group ranking (Figure 3C):
#     1. Recruitment scope      (n_sites, n_us_states, n_us_regions)
#     2. Eligibility constraints
#     3. Trial design attributes
#     4. Protocol narrative text (largest total mass, 384 dims)
#
#   Recommendation priority order follows SHAP target-level
#   importance and per-feature group ranking:
#     Tier 1 — White/Black gaps → recruitment scope recs first
#     Tier 2 — Asian/AIAN gaps  → eligibility recs first
#     Tier 3 — Sex gaps         → eligibility sex setting first
#     Tier 4 — Age gap          → age bounds + visit burden
#     Tier 5 — Trial design completeness (cross-domain)
#
# Prediction keys (TARGET_COLS.pkl):
#   white_pct, black_pct, asian_pct, aian_pct,
#   male_pct, female_pct, age65_pct
#
# Input feature keys (FEATURE_NAMES.pkl, structured subset):
#   eligibility_sex_ALL, eligibility_sex_FEMALE, eligibility_sex_MALE,
#   eligibility_min_age_yrs, eligibility_max_age_yrs,
#   min_age_missing, max_age_missing,
#   n_sites, n_us_states, n_us_regions, study_type_Unknown
# ============================================================

from simpful import FuzzySystem, FuzzySet, LinguisticVariable

from .scoring_constants import (
    DISEASE_PREVALENCE,
    RACE_GROUPS,
    SEX_GROUPS,
    AGE_GROUPS,
)

__version__ = "2.1.0-census-reference"

# ------------------------------------------------------------
# Evidence-based PDRR thresholds
# ------------------------------------------------------------
# Derived from observed PDRR distributions across 1,290 U.S.
# cardiovascular trials in the training dataset, computed against
# Census population share (race/age) and CVD mortality share (sex).
# Higher threshold = more sensitive = fires on smaller gaps.
# Groups with worse historical representation get higher thresholds.

_PDRR_THRESHOLD = {
    # At or above Census parity in typical trial (PDRR typically > 1.0)
    "white_pct":  0.80,
    "male_pct":   0.80,

    # Chronically below Census parity
    "black_pct":  0.90,
    "female_pct": 0.90,

    # Structural absence in majority of trials relative to Census share
    "asian_pct":  0.95,
    "aian_pct":   0.95,
    "age65_pct":  0.95,
}

# Typical PDRR ranges under Census/mortality share references.
# NOTE: These are representative ranges derived from training data
# analysis; exact median values should be recomputed from training
# data against current DISEASE_PREVALENCE constants.
_TYPICAL_PDRR_RANGE = {
    "white_pct":  (1.20, 1.50),   # Over-represented vs Census in most trials
    "black_pct":  (0.40, 1.00),   # Chronically under-represented
    "asian_pct":  (0.00, 0.50),   # Structural absence in most trials
    "aian_pct":   (0.00, 0.10),   # Near-zero in nearly all trials
    "female_pct": (0.80, 1.00),   # Near parity; historically under-enrolled
    "male_pct":   (1.00, 1.20),   # Near or above parity
    "age65_pct":  (0.10, 0.80),   # Structurally under-enrolled vs Census
}

# Human-readable group labels
_GROUP_LABELS = {
    "white_pct":  "White",
    "black_pct":  "Black or African American",
    "asian_pct":  "Asian",
    "aian_pct":   "American Indian or Alaska Native",
    "female_pct": "Female",
    "male_pct":   "Male",
    "age65_pct":  "adults aged 65 and older",
}

# Human-readable reference source labels for recommendations
_REF_SOURCE = {
    "white_pct":  "U.S. Census population share",
    "black_pct":  "U.S. Census population share",
    "asian_pct":  "U.S. Census population share",
    "aian_pct":   "U.S. Census population share",
    "female_pct": "CVD mortality share",
    "male_pct":   "CVD mortality share",
    "age65_pct":  "U.S. Census population share",
}


# ------------------------------------------------------------
# Severity label — mirrors score_pdrr() bands in scoring.py
# ------------------------------------------------------------

def _severity(pdrr):
    if pdrr is None:
        return "unknown"
    diff = abs(pdrr - 1.0)
    if diff <= 0.2:
        return "minor"
    elif diff <= 0.5:
        return "moderate"
    elif diff <= 1.5:
        return "major"
    else:
        return "critical"


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _clamp(x, lo, hi):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, x))


def _pdrr(pred_val, key):
    """
    Compute PDRR for a subgroup using DISEASE_PREVALENCE.
    Matches compute_icer_score() in scoring.py exactly.

    For race and age groups, DISEASE_PREVALENCE contains Census
    population share. For sex groups, it contains CVD mortality share.

    Returns None if pred_val is None or reference is zero.
    """
    denom = DISEASE_PREVALENCE.get(key)
    if pred_val is None or denom is None or denom <= 0:
        return None
    return _clamp(pred_val, 0.0, 1.0) / denom


# ------------------------------------------------------------
# Domain score helpers
# ------------------------------------------------------------

def _infer_domain_scores(preds):
    return {
        "race": preds.get("race_score", 0.0),
        "sex":  preds.get("sex_score",  0.0),
        "age":  preds.get("age_score",  0.0),
    }


def _infer_baseline_scores():
    return {"race": 6.0, "sex": 3.0, "age": 3.0}


# ------------------------------------------------------------
# PUBLIC ENTRY POINT
# ------------------------------------------------------------

def recommend_nf(payload, preds):
    """
    Subgroup-aware, evidence-based neuro-fuzzy recommendations.

    Thresholds are grounded in observed PDRR distributions across
    1,290 U.S. cardiovascular trials, computed against Census
    population share (race/age) and CVD mortality share (sex).
    Groups with worse historical representation receive more
    sensitive thresholds.

    Recommendation priority order follows SHAP feature importance
    from the companion predictive modeling study.

    Parameters
    ----------
    payload : dict
        Trial design inputs. Relevant structured keys:
          eligibility_sex_ALL, eligibility_sex_FEMALE,
          eligibility_sex_MALE, eligibility_min_age_yrs,
          eligibility_max_age_yrs, min_age_missing,
          max_age_missing, n_sites, n_us_states, n_us_regions,
          study_type_Unknown

    preds : dict
        Model outputs. Required keys:
          white_pct, black_pct, asian_pct, aian_pct,
          male_pct, female_pct, age65_pct
        Optional keys:
          icer_score (0-100 scale), race_score, sex_score, age_score

    Returns
    -------
    list[str]
        Recommendations ordered by SHAP-informed priority.
    """
    domain_scores   = _infer_domain_scores(preds)
    baseline_scores = _infer_baseline_scores()

    return _recommend_nf_core(
        payload         = payload,
        preds           = preds,
        domain_scores   = domain_scores,
        baseline_scores = baseline_scores,
    )


# ------------------------------------------------------------
# INTERNAL CORE
# ------------------------------------------------------------

def _recommend_nf_core(payload, preds, domain_scores, baseline_scores):

    # --------------------------------------------------------
    # 1. ICER/CDR -> ACTION INTENSITY (FUZZY)
    #
    # icer_score must be on 0-100 scale.
    # In Main.py: preds_with_score["icer_score"] = (total_score / 21.0) * 100.0
    #
    # Membership functions:
    #   LOW    [0,1] -> [0,1] -> [40,0]   (poor CDR score)
    #   MEDIUM [30,0] -> [50,1] -> [70,0]  (moderate CDR score)
    #   HIGH   [60,0] -> [100,1] -> [100,1] (good CDR score)
    #
    # Rules:
    #   LOW    -> AGGRESSIVE (many recommendations)
    #   MEDIUM -> MODERATE   (some recommendations)
    #   HIGH   -> LIGHT      (few recommendations)
    # --------------------------------------------------------

    icer = _clamp(preds.get("icer_score", 0), 0, 100)

    FS = FuzzySystem()

    FS.add_linguistic_variable("ICER", LinguisticVariable(
        [
            FuzzySet(points=[[0,  1], [0,  1], [40, 0]], term="LOW"),
            FuzzySet(points=[[30, 0], [50, 1], [70, 0]], term="MEDIUM"),
            FuzzySet(points=[[60, 0], [100,1], [100,1]], term="HIGH"),
        ],
        universe_of_discourse=[0, 100],
    ))

    FS.add_linguistic_variable("ACTION", LinguisticVariable(
        [
            FuzzySet(points=[[0, 1], [0, 1], [3, 0]],   term="LIGHT"),
            FuzzySet(points=[[2, 0], [5, 1], [8, 0]],   term="MODERATE"),
            FuzzySet(points=[[7, 0], [10, 1], [10, 1]], term="AGGRESSIVE"),
        ],
        universe_of_discourse=[0, 10],
    ))

    FS.add_rules([
        "IF (ICER IS LOW)    THEN (ACTION IS AGGRESSIVE)",
        "IF (ICER IS MEDIUM) THEN (ACTION IS MODERATE)",
        "IF (ICER IS HIGH)   THEN (ACTION IS LIGHT)",
    ])

    FS.set_variable("ICER", icer)
    action_level = FS.inference()["ACTION"]

    # --------------------------------------------------------
    # 2. PER-SUBGROUP PDRR
    #
    # Computed using DISEASE_PREVALENCE from scoring_constants.py,
    # matching compute_icer_score() in scoring.py exactly.
    # Race/age: divided by Census population share.
    # Sex: divided by CVD mortality share.
    # --------------------------------------------------------

    sg_pdrr = {
        key: _pdrr(preds.get(key), key)
        for key in _PDRR_THRESHOLD
    }

    def _under(key):
        """True if subgroup PDRR is below its evidence-based threshold."""
        p = sg_pdrr.get(key)
        return p is not None and p < _PDRR_THRESHOLD[key]

    # --------------------------------------------------------
    # 3. SUBGROUP-AWARE SHAP-ORDERED RECOMMENDATIONS
    #
    # Tier 1 — White / Black
    #   SHAP: recruitment scope drives these targets
    #   White threshold 0.80: typical PDRR 1.2-1.5 (over-represented
    #     vs Census); fires only on unusual under-enrollment
    #   Black threshold 0.90: typical PDRR 0.4-1.0 (chronically
    #     under-represented vs Census 13.7%)
    #
    # Tier 2 — Asian / AIAN
    #   SHAP: eligibility constraints relatively more important
    #   Both threshold 0.95: structural absence relative to Census
    #     share in majority of historical trials
    #
    # Tier 3 — Sex
    #   SHAP: eligibility_sex_FEMALE is rank-1 individual feature
    #   Female threshold 0.90: near parity vs CVD mortality share
    #     but historically under-enrolled
    #   Male threshold 0.80: typically at or above CVD mortality parity
    #
    # Tier 4 — Age 65+
    #   SHAP: age bounds + recruitment scope both matter
    #   Threshold 0.95: structurally under-enrolled relative to
    #     Census population share (18%) in most trials
    #
    # Tier 5 — Trial design completeness (cross-domain)
    # --------------------------------------------------------

    recs = []

    # ---- Tier 1: White / Black --------------------------------

    white_under = _under("white_pct")
    black_under = _under("black_pct")

    if white_under or black_under:
        recs.append(
            "Expand recruitment scope by increasing the number of "
            "recruiting sites, U.S. states, and Census regions -- "
            "recruitment scope features (n_us_regions, n_us_states, "
            "n_sites) are the strongest structured predictors of "
            "White and Black enrollment composition"
        )

        if black_under:
            sev = _severity(sg_pdrr["black_pct"])
            recs.append(
                f"Add community-based or non-academic recruitment "
                f"locations to improve access for Black or African "
                f"American participants ({sev} under-representation "
                f"relative to U.S. Census population share of 13.7%; "
                f"PDRR = {sg_pdrr['black_pct']:.2f})"
            )
            recs.append(
                "Review exclusion criteria that may disproportionately "
                "affect underrepresented racial groups"
            )

    # ---- Tier 2: Asian / AIAN --------------------------------

    sparse_under = [k for k in ("asian_pct", "aian_pct") if _under(k)]

    if sparse_under:
        under_names = [_GROUP_LABELS[k] for k in sparse_under]
        pdrr_notes  = [
            f"{_GROUP_LABELS[k]} PDRR = {sg_pdrr[k]:.2f} "
            f"(Census population share = "
            f"{DISEASE_PREVALENCE[k]*100:.1f}%)"
            for k in sparse_under
        ]

        recs.append(
            "Review eligibility criteria for implicit restrictions "
            "that may reduce participation among "
            + ", ".join(under_names)
            + " -- eligibility constraints are the primary structured "
            "predictor of enrollment presence for these groups, which "
            "show structural absence in the majority of historical "
            "cardiovascular trials relative to Census population share "
            "(" + "; ".join(pdrr_notes) + ")"
        )
        recs.append(
            "Expand site diversity to include locations with higher "
            "concentrations of "
            + ", ".join(under_names)
            + " populations in their catchment areas"
        )

    # ---- Tier 3: Sex -----------------------------------------

    female_under = _under("female_pct")
    male_under   = _under("male_pct")

    if female_under or male_under:

        elig_sex_all    = payload.get("eligibility_sex_ALL",    0)
        elig_sex_female = payload.get("eligibility_sex_FEMALE", 0)
        elig_sex_male   = payload.get("eligibility_sex_MALE",   0)

        sex_restricted = (
            not elig_sex_all
            and (elig_sex_female or elig_sex_male)
        )

        if sex_restricted:
            recs.append(
                "The sex eligibility setting (eligibility_sex_FEMALE) "
                "is the single strongest structured predictor of "
                "enrollment composition -- review sex-restricted "
                "eligibility criteria and consider expanding to all "
                "sexes where clinically appropriate"
            )

        if female_under:
            sev = _severity(sg_pdrr["female_pct"])
            recs.append(
                f"Ensure recruitment materials are inclusive and "
                f"accessible to female participants; predicted female "
                f"enrollment reflects {sev} under-representation "
                f"relative to cardiovascular mortality share of 52.6% "
                f"(PDRR = {sg_pdrr['female_pct']:.2f})"
            )

    # ---- Tier 4: Age 65+ -------------------------------------

    if _under("age65_pct"):

        max_age     = payload.get("eligibility_max_age_yrs")
        min_age     = payload.get("eligibility_min_age_yrs")
        max_missing = payload.get("max_age_missing", 0)
        min_missing = payload.get("min_age_missing", 0)
        sev         = _severity(sg_pdrr["age65_pct"])

        if max_missing:
            recs.append(
                "Specify a maximum eligible age explicitly -- "
                "max_age_missing is a top-10 structured predictor "
                "and unspecified maximum age is associated with "
                "lower older-adult enrollment"
            )
        elif max_age is not None and _clamp(max_age, 0, 120) < 75:
            recs.append(
                f"Consider increasing the maximum eligible age -- "
                f"eligibility_max_age_yrs is a top-6 structured "
                f"predictor of age65 enrollment; current value may "
                f"exclude participants aged 65 and older "
                f"({sev} under-representation relative to Census "
                f"population share of 18.0%; "
                f"PDRR = {sg_pdrr['age65_pct']:.2f})"
            )

        if min_missing:
            recs.append(
                "Specify a minimum eligible age explicitly -- "
                "min_age_missing is a top-10 structured predictor "
                "and unspecified minimum age reduces prediction "
                "reliability for older-adult enrollment"
            )
        elif min_age is not None and _clamp(min_age, 0, 120) > 50:
            recs.append(
                "Consider lowering the minimum eligible age "
                "where clinically appropriate"
            )

        recs.append(
            "Reduce visit burden or add decentralized options to "
            "support older adult participation -- recruitment scope "
            "features also predict age65 enrollment alongside "
            "eligibility age bounds; adults aged 65 and older are "
            "structurally under-enrolled relative to their Census "
            "population share of 18.0% in most historical "
            "cardiovascular trials"
        )

    # ---- Tier 5: Trial design completeness -------------------

    if payload.get("study_type_Unknown", 0):
        recs.append(
            "Specify study type and design attributes fully -- "
            "study_type_Unknown is a top-10 structured predictor; "
            "underspecified design fields reduce prediction "
            "reliability across all demographic targets"
        )

    # --------------------------------------------------------
    # 4. MODULATE BY ACTION INTENSITY
    #
    # Slicing preserves SHAP priority -- most important recs first.
    # action_level < 3  -> LIGHT      -> 1 recommendation
    # action_level 3-7  -> MODERATE   -> 3 recommendations
    # action_level >= 7 -> AGGRESSIVE -> all recommendations
    # --------------------------------------------------------

    if not recs:
        return ["No major equity-related design changes recommended at this time"]

    if action_level < 3:
        return recs[:1]
    elif action_level < 7:
        return recs[:3]
    else:
        return recs
