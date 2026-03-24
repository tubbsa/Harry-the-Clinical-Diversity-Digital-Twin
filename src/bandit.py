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
#   bandit.py handles: masking, allocation, intervention model,
#     primary purpose, sponsor, phases, planned enrollment
#
#   For each candidate field, a delta score is estimated based on
#   the distance between the current value and the value most
#   commonly associated with higher CDR scores in historical
#   cardiovascular trials. The action with the highest estimated
#   delta is returned as a single readable recommendation.
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
#     planned_enrollment:  >= 200 participants
#
#   These are used as soft targets. A field already at its target
#   value contributes zero delta. Fields far from target contribute
#   larger delta proportional to the estimated CDR impact.
#
# NOTE: Delta values are heuristic estimates. Once training data
#   CDR distributions by design field are computed, replace
#   _FIELD_DELTA with empirically derived expected CDR gains.
# ============================================================

# ------------------------------------------------------------
# Target values associated with higher CDR scores in historical
# high-diversity cardiovascular trials (CDR >= 14/21).
# Fields already at target contribute zero delta.
# ------------------------------------------------------------
_HIGH_DIVERSITY_TARGETS = {
    "allocation":         {"Randomized"},
    "masking":            {"Double", "Triple", "Quadruple"},
    "intervention_model": {"Parallel"},
    "primary_purpose":    {"Treatment", "Prevention"},
    "Phases":             {"Phase 3", "Phase 4"},
    "Sponsor":            {"NIH", "Government"},
}

# Estimated CDR score delta (0-21 scale) for moving a field
# from an off-target to on-target value.
# Replace with empirically derived values from training data.
_FIELD_DELTA = {
    "allocation":         1.2,
    "masking":            0.8,
    "intervention_model": 0.6,
    "primary_purpose":    1.0,
    "Phases":             1.4,
    "Sponsor":            1.1,
    "planned_enrollment": 0.9,
}

# Human-readable recommendation templates for each field.
# {current} and {target} are filled in at runtime.
_REC_TEMPLATE = {
    "allocation": (
        "Consider changing allocation from '{current}' to Randomized -- "
        "randomized trials show higher demographic diversity in historical "
        "cardiovascular trial data (estimated CDR impact: +{delta:.1f} points)"
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
        "'{current}' (estimated CDR impact of phase advancement: "
        "+{delta:.1f} points)"
    ),
    "Sponsor": (
        "NIH- and government-sponsored cardiovascular trials show higher "
        "demographic diversity than industry-sponsored trials in historical "
        "data; current sponsor is '{current}' "
        "(estimated CDR impact: +{delta:.1f} points)"
    ),
    "planned_enrollment": (
        "Increasing planned enrollment above 200 participants is associated "
        "with higher CDR scores in historical cardiovascular trials; "
        "current planned enrollment is {current} "
        "(estimated CDR impact: +{delta:.1f} points)"
    ),
}


def _enrollment_delta(payload: dict) -> float:
    """Return estimated delta for planned enrollment if below 200."""
    try:
        n = float(payload.get("planned_enrollment", 0))
    except (TypeError, ValueError):
        n = 0
    if n < 200:
        # Scale delta by how far below 200 the enrollment is
        shortfall = (200 - n) / 200
        return _FIELD_DELTA["planned_enrollment"] * shortfall
    return 0.0


def _enrollment_rec(payload: dict, delta: float) -> str:
    n = payload.get("planned_enrollment", "unspecified")
    return _REC_TEMPLATE["planned_enrollment"].format(
        current=n,
        delta=delta,
    )


def bandit_optimize(payload: dict, preds: dict) -> list:
    """
    Evaluate trial design fields not covered by nfrules.py and
    return the single highest-impact actionable recommendation.

    Parameters
    ----------
    payload : dict
        Trial design inputs from build_payload(). Relevant keys:
          allocation, masking, intervention_model, primary_purpose,
          Phases, Sponsor, planned_enrollment
    preds : dict
        Model outputs including icer_score (0-100 scale).

    Returns
    -------
    list[str]
        A single-element list containing the highest-impact
        recommendation as a readable string, or an empty list
        if no improvements are identified.
    """
    candidates = []

    # --- Evaluate categorical fields ---
    for field, targets in _HIGH_DIVERSITY_TARGETS.items():
        current = payload.get(field, "Unknown")
        if current in targets:
            continue  # Already at target, no recommendation needed
        delta = _FIELD_DELTA.get(field, 0.0)
        if delta > 0:
            rec = _REC_TEMPLATE[field].format(
                current=current,
                target=", ".join(sorted(targets)),
                delta=delta,
            )
            candidates.append((delta, rec))

    # --- Evaluate planned enrollment ---
    enroll_delta = _enrollment_delta(payload)
    if enroll_delta > 0:
        rec = _enrollment_rec(payload, enroll_delta)
        candidates.append((enroll_delta, rec))

    if not candidates:
        return []

    # Return the single highest-delta recommendation
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_rec = candidates[0]
    return [best_rec]
