# ============================================================
# components/form_inputs.py
# Handles all Streamlit form inputs and returns structured input data.
#
# Session state persistence: all form values are stored in
# st.session_state under the key "form_values" so they survive
# page navigation (e.g. Main → Report → Main).
#
# Reset button: a "Clear form / New scenario" button wipes
# st.session_state["form_values"] and reruns the page,
# restoring all fields to their defaults.
# ============================================================

import streamlit as st

from utils.constants import (
    INCLUSION_OPTIONS, EXCLUSION_OPTIONS, CONDITION_OPTIONS,
    INTERVENTION_OPTIONS, PRIMARY_OUTCOME_OPTIONS,
    SECONDARY_OUTCOME_OPTIONS
)

# ------------------------------------------------------------
# Default values — used on first load and after reset
# ------------------------------------------------------------
_DEFAULTS = {
    "inclusion_sel":      [],
    "exclusion_sel":      [],
    "condition_sel":      [],
    "intervention_sel":   [],
    "primary_sel":        [],
    "secondary_sel":      [],
    "eligibility_sex":    "All",
    "sponsor":            "Industry",
    "collaborators":      "None",
    "phases":             "Phase 3",
    "funder_type":        "Industry",
    "study_type":         "Interventional",
    "allocation":         "Randomized",
    "intervention_model": "Parallel",
    "masking":            "None",
    "primary_purpose":    "Treatment",
    "min_age":            0,
    "max_age":            120,
    "num_us_regions":     1,
    "num_us_states":      1,
    "num_sites":          1,
    "planned_enrollment": 100,
}


def _init_session_state():
    """Seed session state with defaults on first load."""
    if "form_values" not in st.session_state:
        st.session_state["form_values"] = dict(_DEFAULTS)


def _reset_form():
    """Wipe stored form values and rerun to restore defaults."""
    st.session_state["form_values"] = dict(_DEFAULTS)
    st.rerun()


def _s(key):
    """Shorthand: get current persisted value for a form key."""
    return st.session_state["form_values"].get(key, _DEFAULTS[key])


def render_form_and_collect_inputs():
    """
    Renders the full trial-design input form.
    Persists values in st.session_state["form_values"] so they
    survive page navigation.
    Returns a dictionary of values after submission, or None if
    the user has not yet submitted.
    """
    _init_session_state()

    # ------------------------------------------------------------
    # Reset button — outside the form so it fires immediately
    # ------------------------------------------------------------
    if st.button("🔄 Clear form / New scenario", type="secondary"):
        _reset_form()

    with st.form("trial_inputs"):
        st.subheader("Trial Design Inputs")

        # -------------------------
        # Eligibility & content
        # -------------------------
        inclusion_sel = st.multiselect(
            "Inclusion Criteria", INCLUSION_OPTIONS,
            default=_s("inclusion_sel")
        )
        exclusion_sel = st.multiselect(
            "Exclusion Criteria", EXCLUSION_OPTIONS,
            default=_s("exclusion_sel")
        )
        condition_sel = st.multiselect(
            "Conditions", CONDITION_OPTIONS,
            default=_s("condition_sel")
        )
        intervention_sel = st.multiselect(
            "Interventions", INTERVENTION_OPTIONS,
            default=_s("intervention_sel")
        )
        primary_sel = st.multiselect(
            "Primary Outcome Measures", PRIMARY_OUTCOME_OPTIONS,
            default=_s("primary_sel")
        )
        secondary_sel = st.multiselect(
            "Secondary Outcome Measures", SECONDARY_OUTCOME_OPTIONS,
            default=_s("secondary_sel")
        )

        # -------------------------
        # Trial design
        # -------------------------
        eligibility_sex = st.selectbox(
            "Eligibility Sex", ["All", "Male", "Female", "Unknown"],
            index=["All", "Male", "Female", "Unknown"].index(_s("eligibility_sex"))
        )
        sponsor = st.selectbox(
            "Sponsor", ["NIH", "Industry", "Government", "Other", "Unknown"],
            index=["NIH", "Industry", "Government", "Other", "Unknown"].index(_s("sponsor"))
        )
        collaborators = st.selectbox(
            "Collaborators",
            ["None", "Academic", "NIH", "Industry", "Government", "Other", "Unknown"],
            index=["None", "Academic", "NIH", "Industry", "Government", "Other", "Unknown"].index(_s("collaborators"))
        )
        phases = st.selectbox(
            "Phases", ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "N/A", "Unknown"],
            index=["Phase 1", "Phase 2", "Phase 3", "Phase 4", "N/A", "Unknown"].index(_s("phases"))
        )
        funder_type = st.selectbox(
            "Funder Type", ["NIH", "Industry", "Government", "Other", "Unknown"],
            index=["NIH", "Industry", "Government", "Other", "Unknown"].index(_s("funder_type"))
        )
        study_type = st.selectbox(
            "Study Type",
            ["Interventional", "Observational", "Expanded Access", "Unknown"],
            index=["Interventional", "Observational", "Expanded Access", "Unknown"].index(_s("study_type"))
        )
        allocation = st.selectbox(
            "Allocation", ["Randomized", "Non-Randomized", "Unknown"],
            index=["Randomized", "Non-Randomized", "Unknown"].index(_s("allocation"))
        )
        intervention_model = st.selectbox(
            "Intervention Model",
            ["Parallel", "Crossover", "Single Group", "Unknown"],
            index=["Parallel", "Crossover", "Single Group", "Unknown"].index(_s("intervention_model"))
        )
        masking = st.selectbox(
            "Masking",
            ["None", "Single", "Double", "Triple", "Quadruple", "Unknown"],
            index=["None", "Single", "Double", "Triple", "Quadruple", "Unknown"].index(_s("masking"))
        )
        primary_purpose = st.selectbox(
            "Primary Purpose",
            ["Treatment", "Prevention", "Diagnostic", "Screening",
             "Supportive", "Basic", "Health", "Unknown"],
            index=["Treatment", "Prevention", "Diagnostic", "Screening",
                   "Supportive", "Basic", "Health", "Unknown"].index(_s("primary_purpose"))
        )

        # -------------------------
        # Age bounds
        # -------------------------
        min_age = st.number_input(
            "Minimum Age (years)", min_value=0, max_value=120,
            value=_s("min_age")
        )
        max_age = st.number_input(
            "Maximum Age (years)", min_value=0, max_value=120,
            value=_s("max_age")
        )

        # -------------------------
        # Recruitment & Scale
        # -------------------------
        st.subheader("Recruitment & Scale")
        num_us_regions = st.number_input(
            "Number of U.S. Census regions recruiting",
            min_value=1, max_value=4,
            value=_s("num_us_regions"),
            help=(
                "Number of U.S. Census regions with recruiting sites. "
                "The four regions are Northeast, Midwest, South, and West. "
                "Broader regional coverage is associated with greater "
                "demographic diversity in enrollment."
            )
        )
        num_us_states = st.number_input(
            "Number of U.S. states recruiting",
            min_value=1, max_value=50,
            value=_s("num_us_states"),
            help="Estimated number of U.S. states with recruiting sites"
        )
        num_sites = st.number_input(
            "Number of recruiting sites",
            min_value=1, max_value=500,
            value=_s("num_sites"),
            help="Total number of clinical trial sites"
        )
        planned_enrollment = st.number_input(
            "Planned enrollment size",
            min_value=1, max_value=100000,
            value=_s("planned_enrollment"),
            help="Expected total number of enrolled participants"
        )

        submitted = st.form_submit_button("Run Digital Twin")

    if not submitted:
        return None

    # -------------------------
    # Persist submitted values
    # -------------------------
    submitted_values = {
        "inclusion_sel":      inclusion_sel,
        "exclusion_sel":      exclusion_sel,
        "condition_sel":      condition_sel,
        "intervention_sel":   intervention_sel,
        "primary_sel":        primary_sel,
        "secondary_sel":      secondary_sel,
        "eligibility_sex":    eligibility_sex,
        "sponsor":            sponsor,
        "collaborators":      collaborators,
        "phases":             phases,
        "funder_type":        funder_type,
        "study_type":         study_type,
        "allocation":         allocation,
        "intervention_model": intervention_model,
        "masking":            masking,
        "primary_purpose":    primary_purpose,
        "min_age":            min_age,
        "max_age":            max_age,
        "num_us_regions":     num_us_regions,
        "num_us_states":      num_us_states,
        "num_sites":          num_sites,
        "planned_enrollment": planned_enrollment,
    }
    st.session_state["form_values"] = submitted_values

    return submitted_values
