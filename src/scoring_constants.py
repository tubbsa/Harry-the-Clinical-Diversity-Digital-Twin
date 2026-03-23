# ============================================================
# scoring_constants.py — Final ICER Scoring Constants
# ============================================================

# ---------------------------
# POPULATION REFERENCE PROPORTIONS (0–1)
# ---------------------------
# Reference proportions used in PDRR computation.
# These values MUST match the inputs used in charts and scoring.
#
# Reference selection rationale:
#
#   Race and age groups use U.S. Census population share as the
#   parity benchmark. Population-proportional representation was
#   selected rather than disease prevalence or mortality share
#   because prevalence and mortality data may themselves reflect
#   systemic disparities in diagnosis, treatment access, and vital
#   statistics recording. Anchoring equity targets to observed
#   disparate outcomes risks embedding existing inequities into
#   the scoring framework. Population share provides a neutral,
#   unbiased reference that aligns with FDA diversity guidance
#   and NIH inclusion policies.
#
#   Source: U.S. Census Bureau, QuickFacts, 2025.
#   https://www.census.gov/quickfacts/fact/table/US/
#
#   White: non-Hispanic White alone (57.5%) used to match
#   ClinicalTrials.gov race reporting conventions, which record
#   White as a race category separately from Hispanic ethnicity.
#
#   Sex uses CVD mortality share rather than population share.
#   Women are 52.6% of CVD deaths despite being historically
#   underrepresented in trials — this gap between mortality
#   burden and trial enrollment is the core equity signal for
#   the sex domain. Population share (50.5%) would obscure this.
#   Source: Mosca et al., Circulation. 2011;123:1243-1262.
#   https://pmc.ncbi.nlm.nih.gov/articles/PMC4039306/

DISEASE_PREVALENCE = {
    "white_pct": 0.575,   # 57.5% non-Hispanic White alone (Census 2025)
    "black_pct": 0.137,   # 13.7% Black alone (Census 2025)
    "asian_pct": 0.067,   # 6.7%  Asian alone (Census 2025)
    "aian_pct":  0.014,   # 1.4%  American Indian/Alaska Native alone (Census 2025)

    "female_pct": 0.526,  # 52.6% CVD mortality share (Mosca et al. 2011)
    "male_pct":   0.474,  # 47.4% CVD mortality share (Mosca et al. 2011)

    "age65_pct":  0.180,  # 18.0% adults aged ≥65 (Census 2025)
}

# ---------------------------
# SEX BURDEN (CVD MORTALITY SHARE)
# ---------------------------
# Retained for reference and backward compatibility.
# These values are now the primary sex reference in DISEASE_PREVALENCE.
#
# Source:
# Mosca et al., Circulation. 2011;123:1243-1262.
# Women represent 52.6% of CVD deaths in the United States.
# https://pmc.ncbi.nlm.nih.gov/articles/PMC4039306/
SEX_BURDEN_MORTALITY = {
    "female_pct": 0.526,
    "male_pct":   0.474,
}


# ---------------------------
# GROUP DEFINITIONS
# ---------------------------
# RACE DOMAIN — 4 groups only (ICER core without Hispanic)
RACE_GROUPS = [
    "white_pct",
    "black_pct",
    "asian_pct",
    "aian_pct",
]

# SEX DOMAIN
SEX_GROUPS = [
    "female_pct",
    "male_pct",
]

# AGE DOMAIN
AGE_GROUPS = [
    "age65_pct",
]

# ---------------------------
# DOMAIN MAXIMA
# ---------------------------
DOMAIN_MAX = {
    "race": 12,  # 4 groups * max 3 points
    "sex": 6,    # 2 groups * 3 points
    "age": 3,    # 1 group * 3 points
}
