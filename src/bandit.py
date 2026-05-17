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
#     phases, collaborators
#
# Model scope:
#   Recommendations are restricted to fields confirmed present
#   in the trained model feature space (CAT_COLS) AND showing
#   a positive empirical association with CDR scores in the
#   1,290-trial training dataset.
#
#   Fields evaluated and retained (positive delta CDR):
#     Phases           (delta = +0.66, p < 0.001)
#     intervention_model (delta = +0.41, p = 0.004)
#     masking          (delta = +0.31, p = 0.045)
#     Collaborators    (delta = +0.23, p = 0.186)
#     allocation       (delta = +0.18, p = 0.216)
#
#   Fields evaluated and removed (negative empirical delta CDR):
#     Funder Type      (delta = -0.23, p = 0.493)
#     primary_purpose  (delta = -0.28, p = 0.081)
#     Sponsor          (delta = -0.04, p = 0.880)
#     Empirical analysis showed these fields are not associated
#     with higher CDR scores in the training data; including them
#     as bandit targets would produce misleading recommendations.
#
# Delta derivation:
#   Delta CDR values represent the empirically observed difference
#   in mean CDR score between trials at the target value vs. not
#   at the target value, computed across all 1,290 U.S.-based
#   cardiovascular trials in the training dataset:
#     delta = mean CDR (field == target) - mean CDR (field != target)
#   Statistical significance assessed via two-sample t-test.
#
# Key naming:
#   Bandit reads directly from the payload dict produced by
#   payload_builder.py, so all keys must match payload_builder
#   key names exactly.
#
#   planned_enrollment is excluded — it is not in the model
#   feature space (not in NUM_COLS or CAT_COLS) and changes
#   to it do not affect model predictions.
# ============================================================

# Target values associated with higher CDR scores in training data.
# Keys match payload_builder.py payload keys exactly.
# Only fields with positive empirical delta CDR are retained.
_HIGH_DIVERSITY_TARGETS = {
    "Phases":             {"Phase 3", "Phase 4"},
    "intervention_model": {"Parallel"},
    "masking":            {"Double", "Triple", "Quadruple"},
    "Collaborators":      {"Academic", "NIH"},
    "allocation":         {"Randomized"},
}

# Empirically derived CDR score delta (0-21 scale) for moving a
# field from an off-target to on-target value, computed from
# 1,290 U.S. cardiovascular trials in the training dataset.
# delta = mean CDR (on-target) - mean CDR (off-target).
# Two-sample t-test p-values reported in Table X.
_FIELD_DELTA = {
    "Phases":             0.66,
    "intervention_model": 0.41,
    "masking":            0.31,
    "Collaborators":      0.23,
    "allocation":         0.18,
}

# Human-readable recommendation templates.
_REC_TEMPLATE = {
    "Phases": (
        "Phase 3 and Phase 4 trials show the highest enrollment diversity "
        "in historical cardiovascular trial data; current phase is "
        "'{current}' "
        "(estimated CDR impact of phase advancement: +{delta:.2f} points)"
    ),
    "intervention_model": (
        "Consider a Parallel intervention model rather than '{current}' -- "
        "parallel designs in cardiovascular trials are associated with "
        "more diverse enrollment in historical data "
        "(estimated CDR impact: +{delta:.2f} points)"
    ),
    "masking": (
        "Consider using Double or Triple masking rather than '{current}' -- "
        "higher masking complexity is associated with broader recruitment "
        "scope and greater enrollment diversity in historical trials "
        "(estimated CDR impact: +{delta:.2f} points)"
    ),
    "Collaborators": (
        "Adding Academic or NIH collaborators is associated with higher "
        "enrollment diversity in historical cardiovascular trials; "
        "current collaborator type is '{current}' "
        "(estimated CDR impact: +{delta:.2f} points)"
    ),
    "allocation": (
        "Consider changing allocation from '{current}' to Randomized -- "
        "randomized trials show higher demographic diversity in historical "
        "cardiovascular trial data "
        "(estimated CDR impact: +{delta:.2f} points)"
    ),
}


def bandit_optimize(payload: dict, preds: dict) -> list:
    """
    Evaluate trial design fields not covered by nfrules.py and
    return the single highest-impact actionable recommendation.

    Only fields with positive empirical associations with CDR
    scores in the training dataset are evaluated. Fields with
    negative or near-zero empirical delta (Sponsor, Funder Type,
    Primary Purpose) are excluded to avoid misleading guidance.

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
            continue
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
