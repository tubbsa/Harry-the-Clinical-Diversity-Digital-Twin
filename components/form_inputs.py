# ============================================================
# components/form_inputs.py
# Handles all Streamlit form inputs and returns structured input data
# ============================================================
import streamlit as st
# ---- Option Lists ----
from utils.constants import (
    INCLUSION_OPTIONS, EXCLUSION_OPTIONS, CONDITION_OPTIONS,
    INTERVENTION_OPTIONS, PRIMARY_OUTCOME_OPTIONS,
    SECONDARY_OUTCOME_OPTIONS
)
def render_form_and_collect_inputs():
    """
    Renders the full trial-design input form.
    Returns a dictionary of values or None if the user has not submitted.
    """
    with st.form("trial_inputs"):
        st.subheader("Trial Design Inputs")
        # -------------------------
        # Eligibility & content
        # -------------------------
        inclusion_sel = st.multiselect("Inclusion Criteria", INCLUSION_OPTIONS)
        exclusion_sel = st.multiselect("Exclusion Criteria", EXCLUSION_OPTIONS)
        condition_sel = st.multiselect("Conditions", CONDITION_OPTIONS)
        intervention_sel = st.multiselect("Interventions", INTERVENTION_OPTIONS)
        primary_sel = st.multiselect("Primary Outcome Measures", PRIMARY_OUTCOME_OPTIONS)
        secondary_sel = st.multiselect("Secondary Outcome Measures", SECONDARY_OUTCOME_OPTIONS)
        # -------------------------
        # Trial design
        # -------------------------
        eligibility_sex = st.selectbox("Eligibility Sex", ["All", "Male", "Female", "Unknown"])
        sponsor = st.selectbox("Sponsor", ["NIH", "Industry", "Government", "Other", "Unknown"])
        collaborators = st.selectbox(
            "Collaborators",
            ["None", "Academic", "NIH", "Industry", "Government", "Other", "Unknown"]
        )
        phases = st.selectbox("Phases", ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "N/A", "Unknown"])
        funder_type = st.selectbox("Funder Type", ["NIH", "Industry", "Government", "Other", "Unknown"])
        study_type = st.selectbox(
            "Study Type",
            ["Interventional", "Observational", "Expanded Access", "Unknown"]
        )
        allocation = st.selectbox("Allocation", ["Randomized", "Non-Randomized", "Unknown"])
        intervention_model = st.selectbox(
            "Intervention Model",
            ["Parallel", "Crossover", "Single Group", "Unknown"]
        )
        masking = st.selectbox(
            "Masking",
            ["None", "Single", "Double", "Triple", "Quadruple", "Unknown"]
        )
        primary_purpose = st.selectbox(
            "Primary Purpose",
            ["Treatment", "Prevention", "Diagnostic", "Screening",
             "Supportive", "Basic", "Health", "Unknown"]
        )
        # -------------------------
        # Age bounds
        # -------------------------
        min_age = st.number_input("Minimum Age (years)", min_value=0, max_value=120, value=0)
        max_age = st.number_input("Maximum Age (years)", min_value=0, max_value=120, value=120)
        # =====================================================
        #  SHAP-IMPORTANT DESIGN LEVERS
        # =====================================================
        st.subheader("Recruitment & Scale")
        num_us_regions = st.number_input(
            "Number of U.S. Census regions recruiting",
            min_value=1,
            max_value=4,
            value=1,
            help=(
                "Number of U.S. Census regions with recruiting sites. "
                "The four regions are Northeast, Midwest, South, and West. "
                "Broader regional coverage is associated with greater "
                "demographic diversity in enrollment."
            )
        )
        num_us_states = st.number_input(
            "Number of U.S. states recruiting",
            min_value=1,
            max_value=50,
            value=1,
            help="Estimated number of U.S. states with recruiting sites"
        )
        num_sites = st.number_input(
            "Number of recruiting sites",
            min_value=1,
            max_value=500,
            value=1,
            help="Total number of clinical trial sites"
        )
        planned_enrollment = st.number_input(
            "Planned enrollment size",
            min_value=1,
            max_value=100000,
            value=100,
            help="Expected total number of enrolled participants"
        )
        submitted = st.form_submit_button("Run Digital Twin")
    if not submitted:
        return None
    # -------------------------
    # Return structured input data
    # -------------------------
    return {
        "inclusion_sel": inclusion_sel,
        "exclusion_sel": exclusion_sel,
        "condition_sel": condition_sel,
        "intervention_sel": intervention_sel,
        "primary_sel": primary_sel,
        "secondary_sel": secondary_sel,
        "eligibility_sex": eligibility_sex,
        "sponsor": sponsor,
        "collaborators": collaborators,
        "phases": phases,
        "funder_type": funder_type,
        "study_type": study_type,
        "allocation": allocation,
        "intervention_model": intervention_model,
        "masking": masking,
        "primary_purpose": primary_purpose,
        "min_age": min_age,
        "max_age": max_age,
        # SHAP-aligned recruitment inputs
        "num_us_regions": num_us_regions,
        "num_us_states": num_us_states,
        "num_sites": num_sites,
        "planned_enrollment": planned_enrollment,
    }

