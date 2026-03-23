# ============================================================
# rep_prev_diverging.py — Δ (Trial − Population Reference) Chart
# CLEAN, SORTED, DASHBOARD-READY
# ============================================================
import plotly.graph_objects as go
from utils.constants import DISPLAY_LABELS

# Order for visualization (used as filter, not final order)
ORDERED_GROUPS = [
    "white_pct",
    "black_pct",
    "asian_pct",
    "aian_pct",
    "female_pct",
    "male_pct",
    "age65_pct",
]


def make_rep_prev_diverging(preds_frac: dict, population_ref: dict):
    """
    Create a diverging bar chart showing:
        Delta = Trial enrollment - Population reference (percentage points)
    - Uses fractions (0-1) internally
    - Displays signed percentage-point differences
    - Skips groups with missing data
    - Sorts by absolute Delta (largest gaps first)
    - Orange = under-represented (below population reference)
    - Green = over-represented (above population reference)
    """
    rows = []
    for g in ORDERED_GROUPS:
        pred = preds_frac.get(g)
        ref  = population_ref.get(g)
        if pred is None or ref is None:
            continue
        delta_pp = (pred - ref) * 100
        rows.append(
            {
                "group": DISPLAY_LABELS.get(g, g),
                "delta": delta_pp,
                "color": "#009E73" if delta_pp >= 0 else "#D55E00",
            }
        )

    # Sort by magnitude of gap
    rows = sorted(rows, key=lambda r: abs(r["delta"]), reverse=True)

    labels = [r["group"] for r in rows]
    deltas = [r["delta"] for r in rows]
    colors = [r["color"] for r in rows]

    fig = go.Figure()
    fig.add_bar(
        y=labels,
        x=deltas,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f} pp" for v in deltas],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Enrollment vs. population reference: %{x:+.1f} pp"
            "<extra></extra>"
        ),
        name="Delta (Trial - Population Reference)",
    )

    fig.add_vline(
        x=0,
        line_width=2,
        line_dash="dash",
        line_color="#333333",
    )

    fig.update_layout(
        title=dict(
            text="Representation vs Population Reference (Delta)",
            font=dict(size=20),
        ),
        xaxis=dict(
            title="Difference (percentage points)",
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=12),
        ),
        showlegend=False,
        height=380,
        margin=dict(l=120, r=40, t=70, b=40),
        annotations=[
            dict(
                text="Positive values indicate over-representation relative to U.S. population share; negative values indicate under-representation.",
                xref="paper",
                yref="paper",
                x=0,
                y=1.08,
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )
    return fig
