# ============================================================
# FILE: src/bandit.py
# ============================================================
# Purpose:
#   Complement the neuro-fuzzy recommendation layer (nfrules.py)
#   by evaluating trial design fields that nfrules does not cover
#   and recommending the single highest-impact change available.
#
# Design:
#   nfrules.py handles: eligibility sex, age bounds, recruitment
#     scope (n_sites, n_us_states, n_us_regions), study type
#   bandit.py handles: allocation, masking, intervention model,
#     primary purpose, sponsor, collaborators, phases, funder type
#
#   For each candidate field, a delta score is estimated based on
#   the distance between the current value and the value most
#   commonly associated with higher CDR scores in historical
#   cardiovascular trials. The action with the highest estimated
#   delta is returned as a single readable recommendation.
#
# Model scope:
#   Recommendations are restricted to fields confirmed present
#   in the trained model feature space (CAT_COLS):
#     eligibility_sex, sponsor, collaborators, phases,
#     funder_type, study_type, allocation, intervention_model,
#     masking, primary_purpose
#
#   Fields NOT in the model feature space and therefore excluded:
#     planned_enrollment — not in NUM_COLS or CAT_COLS;
#       changes to this field do not affect model predictions.
#
#   nfrules.py covers eligibility_sex, study_type, and
#   recruitment scope, so bandit.py evaluates the remainder.
#
# Evidence basis:
#   High-diversity cardiovascular trials in the training dataset
#   (CDR >= 14/21) show the following modal design patterns:
#     allocation:          Randomized
#     masking:             Double or Triple
#     intervention_model:  Parallel
#     primary_purpose:     Treatment or Prevention
#     phases:              Phase 3 or Phase 4
#     sponsor:             NIH or Government
#     funder_type:         NIH or Government
#     collaborators:       Academic or NIH
#
#   These are used as soft targets. A field already at its target
#   value contributes zero delta. Fields far from target contribute
#   larger delta proportional to the estimated CDR impact.
#
# NOTE: Delta values are heuristic estimates derived from modal
#   design patterns in high-CDR training trials. Replace
#   _FIELD_DELTA with empirically derived expected CDR gains
#   once CDR distributions by design field are computed from
#   training data.
# ============================================================

# ------------------------------------------------------------
# Payload key names used by payload_builder.py for each field.
# These are the keys bandit.py reads from the payload dict.
# Note: predictor.py remaps these to CAT_COLS names internally;
# bandit.py reads from the payload directly so uses payload keys.
# ------------------------------------------------------------

# Target values associated with higher CDR scores.
# Fields already at their target contribute zero delta.
_HIGH_DIVERSITY_TARGETS = {
    "allocation":         {"Randomized"},
    "masking":            {"Double", "Triple", "Quadruple"},
    "intervention_model": {"Parallel"},
    "primary_purpose":    {"Treatment", "Prevention"},
    "Phases":             {"Phase 3", "Phase 4"},
    "Sponsor":            {"NIH", "Government"},
    "funder_type":        {"NIH", "Government"},
    "collaborators":      {"Academic", "NIH"},
}

# Estimated CDR score delta (0-21 scale) for moving a field
# from an off-target to on-target value.
# All fields listed here are confirmed present in the model
# feature space (CAT_COLS). planned_enrollment is excluded
# because it is not in the model feature space.
_FIELD_DELTA = {
    "allocation":         1.2,
    "masking":            0.8,
    "intervention_model": 0.6,
    "primary_purpose":    1.0,
    "Phases":             1.4,
    "Sponsor":            1.1,
    "funder_type":        0.9,
    "collaborators":      0.7,
}

# Human-readable recommendation templates.
# {current} is filled with the trial's current value at runtime.
# {delta} is filled with the estimated CDR impact.
_REC_TEMPLATE = {
    "allocation": (
        "Consider changing allocation from '{current}' to Randomized -- "
        "randomized trials show higher demographic diversity in historical "
        "cardiovascular trial data "
        "(estimated CDR impact: +{delta:.1f} points)"
    ),
    "masking": (
        "Consider using Double or Triple masking rather than '{current}' -- "
        "higher masking complexity is associated with broader recruitment "
        "scope and greater enrollment diversity in historical trials "
        "(estimated CDR impact: +{delta:.1f} points)"
    ),
    "intervention_model": (
        "Consider a Parallel intervention model rather than '{current}' -- "
        "parallel designs in cardiovascular trials are associated with "
        "more diverse enrollment in historical data "
        "(estimated CDR impact: +{delta:.1f} points)"
    ),
    "primary_purpose": (
        "Consider reframing primary purpose as Treatment or Prevention "
        "rather than '{current}' where clinically appropriate -- trials "
        "with these purposes show higher demographic diversity in "
        "historical cardiovascular data "
        "(estimated CDR impact: +{delta:.1f} points)"
    ),
    "Phases": (
        "Phase 3 and Phase 4 trials show the highest enrollment diversity "
        "in historical cardiovascular trial data; current phase is "
        "'{current}' "
        "(estimated CDR impact of phase advancement: +{delta:.1f} points)"
    ),
    "Sponsor": (
        "NIH- and government-sponsored cardiovascular trials show higher "
        "demographic diversity than industry-sponsored trials in historical "
        "data; current sponsor is '{current}' "
        "(estimated CDR impact: +{delta:.1f} points)"
    ),
    "funder_type": (
        "Trials funded by NIH or government sources show higher demographic "
        "diversity than industry-funded trials in historical cardiovascular "
        "data; current funder type is '{current}' "
        "(estimated CDR impact: +{delta:.1f} points)"
    ),
    "collaborators": (
        "Adding Academic or NIH collaborators is associated with higher "
        "enrollment diversity in historical cardiovascular trials; "
        "current collaborator type is '{current}' "
        "(estimated CDR impact: +{delta:.1f} points)"
    ),
}


def bandit_optimize(payload: dict, preds: dict) -> list:
    """
    Evaluate trial design fields not covered by nfrules.py and
    return the single highest-impact actionable recommendation.

    Only fields confirmed present in the model feature space
    (CAT_COLS) are evaluated. planned_enrollment is excluded
    because it is not in the model feature space and changes
    to it do not affect model predictions.

    Parameters
    ----------
    payload : dict
        Trial design inputs from build_payload(). Relevant keys:
          allocation, masking, intervention_model, primary_purpose,
          Phases, Sponsor, funder_type, collaborators
    preds : dict
        Model outputs including predicted demographic proportions
        and icer_score.

    Returns
    -------
    list[str]
        A single-element list containing the highest-impact
        recommendation as a readable string, or an empty list
        if no improvements are identified (all fields at target).
    """
    candidates = []

    for field, targets in _HIGH_DIVERSITY_TARGETS.items():
        current = payload.get(field, "Unknown")
        if current in targets:
            continue  # Already at target, no recommendation needed
        delta = _FIELD_DELTA.get(field, 0.0)
        if delta > 0:
            rec = _REC_TEMPLATE[field].format(
                current=current,
                delta=delta,
            )
            candidates.append((delta, rec))

    if not candidates:
        return []

    # Return the single highest-delta recommendation
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_rec = candidates[0]
    return [best_rec]
