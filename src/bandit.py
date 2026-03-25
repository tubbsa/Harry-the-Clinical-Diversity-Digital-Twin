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
# Model scope:
#   Recommendations are restricted to fields confirmed present
#   in the trained model feature space (CAT_COLS):
#     eligibility_sex, sponsor, collaborators, phases,
#     funder_type, study_type, allocation, intervention_model,
#     masking, primary_purpose
#
#   planned_enrollment is excluded — it is not in the model
#   feature space (not in NUM_COLS or CAT_COLS) and changes
#   to it do not affect model predictions.
#
# Key naming:
#   Bandit reads directly from the payload dict produced by
#   payload_builder.py, so all keys must match payload_builder
#   key names exactly:
#     "Sponsor"       (payload_builder key; CAT_COL: sponsor)
#     "Collaborators" (payload_builder key; CAT_COL: collaborators)
#     "Phases"        (payload_builder key; CAT_COL: phases)
#     "Funder Type"   (payload_builder key; CAT_COL: funder_type)
#   allocation, intervention_model, masking, primary_purpose
#   are identical between payload_builder and CAT_COLS.
#
# Evidence basis:
#   High-diversity cardiovascular trials in the training dataset
#   (CDR >= 14/21) show the following modal design patterns:
#     allocation:          Randomized
#     masking:             Double or Triple
#     intervention_model:  Parallel
#     primary_purpose:     Treatment or Prevention
#     Phases:              Phase 3 or Phase 4
#     Sponsor:             NIH or Government
#     Funder Type:         NIH or Government
#     Collaborators:       Academic or NIH
#
# NOTE: Delta values are heuristic estimates. Replace
#   _FIELD_DELTA with empirically derived expected CDR gains
#   once CDR distributions by design field are computed from
#   training data.
# ============================================================

# Target values associated with higher CDR scores.
# Keys match payload_builder.py payload keys exactly.
_HIGH_DIVERSITY_TARGETS = {
    "allocation":         {"Randomized"},
    "masking":            {"Double", "Triple", "Quadruple"},
    "intervention_model": {"Parallel"},
    "primary_purpose":    {"Treatment", "Prevention"},
    "Phases":             {"Phase 3", "Phase 4"},
    "Sponsor":            {"NIH", "Government"},
    "Funder Type":        {"NIH", "Government"},
    "Collaborators":      {"Academic", "NIH"},
}

# Estimated CDR score delta (0-21 scale) for moving a field
# from an off-target to on-target value.
# All fields are confirmed present in CAT_COLS.
# planned_enrollment is excluded — not in model feature space.
_FIELD_DELTA = {
    "allocation":         1.2,
    "masking":            0.8,
    "intervention_model": 0.6,
    "primary_purpose":    1.0,
    "Phases":             1.4,
    "Sponsor":            1.1,
    "Funder Type":        0.9,
    "Collaborators":      0.7,
}

# Human-readable recommendation templates.
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
    "Funder Type": (
        "Trials funded by NIH or government sources show higher demographic "
        "diversity than industry-funded trials in historical cardiovascular "
        "data; current funder type is '{current}' "
        "(estimated CDR impact: +{delta:.1f} points)"
    ),
    "Collaborators": (
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
    because it is not in the model feature space.

    Parameters
    ----------
    payload : dict
        Trial design inputs from build_payload(). Uses
        payload_builder.py key names exactly.
    preds : dict
        Model outputs including predicted demographic proportions.

    Returns
    -------
    list[str]
        A single-element list containing the highest-impact
        recommendation, or an empty list if all fields are
        already at their target values.
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
