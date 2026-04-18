# ============================================================
# components/score_tiles.py
# Renders Race / Gender / Age / Total ICER metric tiles
# + OOD reliability flag
# ============================================================
import streamlit as st
from utils.constants import DISPLAY_LABELS

def label_domain(score, max_score):
    pct = score / max_score
    if pct >= 0.80:
        return "Good", "metric-good"
    elif pct >= 0.60:
        return "Fair", "metric-fair"
    else:
        return "Poor", "metric-poor"

def render_score_tiles(total_score, breakdown, ood_score=None, is_ood=False):
    """Render the four domain score tiles based on ICER scoring results,
    plus an OOD reliability flag if ood_score is provided."""

    race_score = breakdown[breakdown["domain"]=="race"]["score"].sum()
    sex_score  = breakdown[breakdown["domain"]=="sex"]["score"].sum()
    age_score  = breakdown[breakdown["domain"]=="age"]["score"].sum()

    total_label, total_style = label_domain(total_score, 21)
    race_label, race_style   = label_domain(race_score, 12)
    sex_label, sex_style     = label_domain(sex_score, 6)
    age_label, age_style     = label_domain(age_score, 3)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="{total_style}"><strong>Total Score:</strong> '
            f'{total_score:.1f} / 21<br><em>{total_label}</em></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="{race_style}"><strong>Race:</strong> '
            f'{race_score:.1f} / 12<br><em>{race_label}</em></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="{sex_style}"><strong>Sex:</strong> '
            f'{sex_score:.1f} / 6<br><em>{sex_label}</em></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="{age_style}"><strong>Age:</strong> '
            f'{age_score:.1f} / 3<br><em>{age_label}</em></div>',
            unsafe_allow_html=True,
        )

    # OOD reliability flag
    if ood_score is not None:
        st.markdown("---")
        if is_ood:
            st.warning(
                f"⚠️ **Projection Reliability: Low** — This trial design is outside the "
                f"model's training distribution (OOD score: {ood_score:.2f}). "
                f"Predictions are generated but should be interpreted with caution. "
                f"This design may differ substantially from trials the model was trained on."
            )
        else:
            st.success(
                f"✅ **Projection Reliability: High** — This trial design is within the "
                f"model's training distribution (OOD score: {ood_score:.2f}). "
                f"Predictions have strong empirical support."
            )
