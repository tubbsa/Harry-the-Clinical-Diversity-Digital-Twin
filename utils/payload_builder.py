# ============================================================
# payload_builder.py — Convert Streamlit form output into
# model-ready payload dict for predictor.py
# ============================================================
def build_payload(inputs: dict) -> dict:
    """
    Convert the dictionary returned by render_form_and_collect_inputs()
    into a payload dict with the exact fields expected by predictor.py.

    SHAP-important recruitment fields:
      n_us_regions  — top-2 structured predictor (Figure 3B)
      n_us_states   — top-3 structured predictor
      n_sites       — top-8 structured predictor
    These must be passed through to the model for recommendations
    to have any effect on predicted enrollment proportions.
    """
    # ---------------------------
    # TEXT FIELDS (SEPARATE, NOT MERGED)
    # ---------------------------
    inclusion_text     = "; ".join(inputs.get("inclusion_sel", []))
    exclusion_text     = "; ".join(inputs.get("exclusion_sel", []))
    condition_text     = "; ".join(inputs.get("condition_sel", []))
    intervention_text  = "; ".join(inputs.get("intervention_sel", []))
    primary_text       = "; ".join(inputs.get("primary_sel", []))
    secondary_text     = "; ".join(inputs.get("secondary_sel", []))

    # ---------------------------
    # NUMERIC FIELDS
    # ---------------------------
    min_age = inputs.get("min_age", 0)
    max_age = inputs.get("max_age", 120)

    # Age missingness flags — used as top-10 structured predictors
    min_age_missing = 1 if min_age == 0 else 0
    max_age_missing = 1 if max_age == 120 else 0

    # Recruitment scope — top SHAP structured predictors
    # Form keys (num_*) mapped to model feature names (n_*)
    n_us_regions = inputs.get("num_us_regions", 1)
    n_us_states  = inputs.get("num_us_states",  1)
    n_sites      = inputs.get("num_sites",      1)

    # Study type unknown flag
    study_type_raw     = inputs.get("study_type", "Unknown")
    study_type_unknown = 1 if study_type_raw == "Unknown" else 0

    # Sex eligibility one-hot flags (top SHAP features)
    eligibility_sex_raw    = inputs.get("eligibility_sex", "Unknown")
    eligibility_sex_all    = 1 if eligibility_sex_raw == "All"     else 0
    eligibility_sex_female = 1 if eligibility_sex_raw == "Female"  else 0
    eligibility_sex_male   = 1 if eligibility_sex_raw == "Male"    else 0

    payload = {
        # TEXT FIELDS (exact names predictor.py expects)
        "inclusion_text":              inclusion_text,
        "exclusion_text":              exclusion_text,
        "Conditions":                  condition_text,
        "Interventions":               intervention_text,
        "Primary Outcome Measures":    primary_text,
        "Secondary Outcome Measures":  secondary_text,

        # AGE BOUNDS + MISSINGNESS FLAGS
        "eligibility_min_age":         min_age,
        "eligibility_max_age":         max_age,
        "eligibility_min_age_yrs":     min_age,
        "eligibility_max_age_yrs":     max_age,
        "min_age_missing":             min_age_missing,
        "max_age_missing":             max_age_missing,

        # SEX ELIGIBILITY — raw value and one-hot flags
        "eligibility_sex":             eligibility_sex_raw,
        "eligibility_sex_ALL":         eligibility_sex_all,
        "eligibility_sex_FEMALE":      eligibility_sex_female,
        "eligibility_sex_MALE":        eligibility_sex_male,

        # RECRUITMENT SCOPE (SHAP top-2, top-3, top-8)
        "n_us_regions":                n_us_regions,
        "n_us_states":                 n_us_states,
        "n_sites":                     n_sites,

        # STUDY TYPE
        "Study Type":                  study_type_raw,
        "study_type_Unknown":          study_type_unknown,

        # OTHER CATEGORICAL FIELDS
        "Sponsor":                     inputs.get("sponsor",           "Unknown"),
        "Collaborators":               inputs.get("collaborators",     "Unknown"),
        "Phases":                      inputs.get("phases",            "Unknown"),
        "Funder Type":                 inputs.get("funder_type",       "Unknown"),
        "allocation":                  inputs.get("allocation",        "Unknown"),
        "intervention_model":          inputs.get("intervention_model","Unknown"),
        "masking":                     inputs.get("masking",           "Unknown"),
        "primary_purpose":             inputs.get("primary_purpose",   "Unknown"),

        # PLANNED ENROLLMENT
        "planned_enrollment":          inputs.get("planned_enrollment", 100),

        # STATIC KEYS predictor.py expects
        "country":                     "United States",
        "continent":                   "North America",
    }
    return payload
