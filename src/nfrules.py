# ============================================================
# nfrules.py — Neuro-Fuzzy Policy Layer (SUBGROUP-AWARE)
# Abigail Tubbs
#
# Purpose:
#   Translate model-predicted ICER + subgroup proportions into
#   interpretable, actionable trial design recommendations.
#
# Design:
#   • CatBoost     = learning
#   • ICER/CDR     = scoring
#   • Fuzzy system = policy interpretation (NOT trained)
#
# Gap metric:
#   All gaps are expressed as PDRR = trial_val / disease_prevalence,
#   matching the scoring logic in scoring.py / score_pdrr().
#   PDRR = 1.0 means perfect parity with disease burden.
#   PDRR < 1.0 means under-represented relative to disease burden.
#   Recommendations trigger when PDRR falls below a threshold.
#
# Threshold rationale — grounded in observed PDRR distributions
# across 1,290 U.S. cardiovascular trials (training dataset):
#
#   Group        Median PDRR   % trials < 0.8   % trials = 0
#   --------------------------------------------------------
#   White        1.032         27.8%             8.6%
#   Male         1.152         21.5%             6.4%
#   Black        0.690         54.4%            25.1%
#   Female       0.850         45.0%             5.2%
#   Asian        0.000         76.7%            54.3%
#   AIAN         0.000         93.5%            83.9%
#   Age65        0.000         78.9%            75.2%
#
#   Threshold assignment:
#     White / Male   — median PDRR >= 1.0 (at or above parity)
#                      threshold 0.80: fire when meaningfully below parity
#
#     Black / Female — median PDRR < 1.0 (chronic under-representation)
#                      threshold 0.90: fire earlier to catch gaps sooner
#
#     Asian / AIAN / Age65 — median PDRR = 0.0 (structural absence
#                      in majority of trials)
#                      threshold 0.95: most sensitive, fire at any
#                      meaningful departure from parity
#
#   Principle: the worse the historical representation, the higher
#   the threshold and the more sensitive the recommendation trigger.
#   This grounds differential sensitivity in observed evidence rather
#   than arbitrary design choices.
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

__version__ = "2.0.0-shap-evidence-based"
# ------------------------------------------------------------
# Evidence-based PDRR thresholds
# ------------------------------------------------------------
# Derived from observed PDRR distributions across 1,290 U.S.
# cardiovascular trials in the training dataset.
# Higher threshold = more sensitive = fires on smaller gaps.
# Groups with worse historical representation get higher thresholds.

_PDRR_THRESHOLD = {
    # At or above parity in typical trial (median PDRR >= 1.0)
    "white_pct":  0.80,
    "male_pct":   0.80,

    # Chronically below parity (median PDRR < 1.0)
    "black_pct":  0.90,
    "female_pct": 0.90,

    # Structural absence in majority of trials (median PDRR = 0.0)
    "asian_pct":  0.95,
    "aian_pct":   0.95,
    "age65_pct":  0.95,
}

# Observed median PDRRs from training data (for reference/reporting)
_OBSERVED_MEDIAN_PDRR = {
    "white_pct":  1.032,
    "black_pct":  0.690,
    "asian_pct":  0.000,
    "aian_pct":   0.000,
    "female_pct": 0.850,
    "male_pct":   1.152,
    "age65_pct":  0.000,
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
    Returns None if pred_val is None or prevalence is zero.
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
    1,290 U.S. cardiovascular trials. Groups with worse historical
    representation receive more sensitive thresholds.

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
    # 1. ICER → ACTION INTENSITY (FUZZY)
    #
    # icer_score must be on 0-100 scale.
    # In Main.py: preds_with_score["icer_score"] = (total_score / 21.0) * 100.0
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
    #   Thresholds: White 0.80 (median PDRR 1.03),
    #               Black 0.90 (median PDRR 0.69)
    #
    # Tier 2 — Asian / AIAN
    #   SHAP: eligibility constraints relatively more important
    #   Thresholds: both 0.95 (median PDRR 0.00)
    #
    # Tier 3 — Sex
    #   SHAP: eligibility_sex_FEMALE is rank-1 individual feature
    #   Thresholds: Female 0.90 (median PDRR 0.85),
    #               Male   0.80 (median PDRR 1.15)
    #
    # Tier 4 — Age ≥65
    #   SHAP: age bounds + recruitment scope both matter
    #   Threshold: 0.95 (median PDRR 0.00)
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
            "recruiting sites, U.S. states, and Census regions — "
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
                f"relative to cardiovascular disease burden; "
                f"PDRR = {sg_pdrr['black_pct']:.2f}, historical "
                f"median PDRR = {_OBSERVED_MEDIAN_PDRR['black_pct']:.3f})"
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
            f"(historical median = {_OBSERVED_MEDIAN_PDRR[k]:.3f})"
            for k in sparse_under
        ]

        recs.append(
            "Review eligibility criteria for implicit restrictions "
            "that may reduce participation among "
            + ", ".join(under_names)
            + " — eligibility constraints are the primary structured "
            "predictor of enrollment presence for these groups, which "
            "show structural absence in the majority of historical "
            "cardiovascular trials (" + "; ".join(pdrr_notes) + ")"
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
                "enrollment composition — review sex-restricted "
                "eligibility criteria and consider expanding to all "
                "sexes where clinically appropriate"
            )

        if female_under:
            sev = _severity(sg_pdrr["female_pct"])
            recs.append(
                f"Ensure recruitment materials are inclusive and "
                f"accessible across genders; predicted female "
                f"enrollment reflects {sev} under-representation "
                f"relative to cardiovascular disease burden "
                f"(PDRR = {sg_pdrr['female_pct']:.2f}, historical "
                f"median PDRR = {_OBSERVED_MEDIAN_PDRR['female_pct']:.3f})"
            )

    # ---- Tier 4: Age ≥65 -------------------------------------

    if _under("age65_pct"):

        max_age     = payload.get("eligibility_max_age_yrs")
        min_age     = payload.get("eligibility_min_age_yrs")
        max_missing = payload.get("max_age_missing", 0)
        min_missing = payload.get("min_age_missing", 0)
        sev         = _severity(sg_pdrr["age65_pct"])

        if max_missing:
            recs.append(
                "Specify a maximum eligible age explicitly — "
                "max_age_missing is a top-10 structured predictor "
                "and unspecified maximum age is associated with "
                "lower older-adult enrollment"
            )
        elif max_age is not None and _clamp(max_age, 0, 120) < 75:
            recs.append(
                f"Consider increasing the maximum eligible age — "
                f"eligibility_max_age_yrs is a top-6 structured "
                f"predictor of age65 enrollment; current value may "
                f"exclude participants aged 65 and older "
                f"({sev} under-representation; "
                f"PDRR = {sg_pdrr['age65_pct']:.2f}, historical "
                f"median PDRR = {_OBSERVED_MEDIAN_PDRR['age65_pct']:.3f})"
            )

        if min_missing:
            recs.append(
                "Specify a minimum eligible age explicitly — "
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
            "support older adult participation — recruitment scope "
            "features also predict age65 enrollment alongside "
            "eligibility age bounds; adults aged 65 and older show "
            f"structural absence in {78.9:.1f}% of historical "
            "cardiovascular trials"
        )

    # ---- Tier 5: Trial design completeness -------------------

    if payload.get("study_type_Unknown", 0):
        recs.append(
            "Specify study type and design attributes fully — "
            "study_type_Unknown is a top-10 structured predictor; "
            "underspecified design fields reduce prediction "
            "reliability across all demographic targets"
        )

    # --------------------------------------------------------
    # 4. MODULATE BY ACTION INTENSITY
    #
    # Slicing preserves SHAP priority — most important recs first.
    # --------------------------------------------------------

    if not recs:
        return ["No major equity-related design changes recommended at this time"]

    if action_level < 3:
        return recs[:1]      # LIGHT
    elif action_level < 7:
        return recs[:3]      # MODERATE
    else:
        return recs          # AGGRESSIVE
