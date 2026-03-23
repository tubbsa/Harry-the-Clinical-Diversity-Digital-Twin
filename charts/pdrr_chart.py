# ============================================================
# pdrr_chart.py — PDRR bar chart (Enrollment vs Population Reference)
# DASHBOARD-READY, NONE-SAFE
# ============================================================

from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from utils.constants import DISPLAY_LABELS
except Exception:
    DISPLAY_LABELS = {}

# -----------------------------
# Display-only label mappings
# -----------------------------
RENAME_MAP = {
    "white_pct": "White %",
    "black_pct": "Black %",
    "asian_pct": "Asian %",
    "aian_pct": "American Indian / Alaska Native %",
    "female_pct": "Female %",
    "male_pct": "Male %",
    "age65_pct": "Age 65+ %",
}

DEMO_MAP = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "American Indian / Alaska Native",
    4: "Female",
    5: "Male",
    6: "Age 65+",
}

# Reference column name — updated from disease_prevalence to population_reference
REFERENCE_COL_CANDIDATES = [
    "population_reference",
    "disease_prevalence",   # backward compat
    "prev_frac",
    "prevalence",
    "disease_prev",
    "prev_pct",
]


def _to_dataframe(breakdown: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
    if isinstance(breakdown, pd.DataFrame):
        return breakdown.copy()
    elif isinstance(breakdown, dict):
        if breakdown and isinstance(next(iter(breakdown.values())), dict):
            df = pd.DataFrame.from_dict(breakdown, orient="index")
            df.index.name = "group"
            return df.reset_index()
        return pd.DataFrame([breakdown])
    else:
        raise TypeError("breakdown must be a dict or pandas.DataFrame")


def _find_group_column(df: pd.DataFrame) -> str:
    for cand in ["component", "group", "subgroup", "category", "label"]:
        if cand in df.columns:
            return cand
    df["group"] = df.index.astype(str)
    return "group"


def _find_reference_column(df: pd.DataFrame) -> str | None:
    for cand in REFERENCE_COL_CANDIDATES:
        if cand in df.columns:
            return cand
    return None


def _find_value_column(df: pd.DataFrame) -> str | None:
    for cand in ["value", "trial_frac", "trial", "pred_frac", "predicted", "trial_pct"]:
        if cand in df.columns:
            return cand
    return None


def _ensure_pdrr_column(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["pdrr", "PDRR", "pdr_ratio", "pdrr_ratio"]:
        if c in df.columns:
            df["pdrr"] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            return df

    trial_col = _find_value_column(df)
    prev_col  = _find_reference_column(df)

    if trial_col and prev_col:
        df["pdrr"] = (
            pd.to_numeric(df[trial_col], errors="coerce")
            / pd.to_numeric(df[prev_col], errors="coerce").replace(0, np.nan)
        ).fillna(0.0)
        return df

    df["pdrr"] = 1.0
    return df


def _parse_pct_string(x: Any) -> float:
    """Convert '74.8%' or 0.748 or None to a 0-1 float."""
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        v = float(x)
        # If already a proportion (0-1 range) leave it; if >1 assume already percentage
        return v / 100.0 if v > 1.5 else v
    s = str(x).strip().rstrip("%")
    try:
        v = float(s)
        return v / 100.0 if v > 1.5 else v
    except ValueError:
        return 0.0


def _coerce_int_like(x: Any) -> Any:
    if x is None:
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float) and np.isfinite(x) and float(x).is_integer():
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if s.isdigit():
            try:
                return int(s)
            except Exception:
                return x
    return x


def make_pdrr_bar_chart(
    breakdown: Union[pd.DataFrame, Dict[str, Any]],
    threshold: float = 1.0,
) -> go.Figure:
    df = _to_dataframe(breakdown)
    group_col = _find_group_column(df)
    df = _ensure_pdrr_column(df)

    # Find value and reference columns
    value_col = _find_value_column(df)
    ref_col   = _find_reference_column(df)

    # -----------------------------
    # Human-readable labels
    # -----------------------------
    def prettify_label(g: Any) -> str:
        if g in DISPLAY_LABELS:
            return str(DISPLAY_LABELS[g])
        gi = _coerce_int_like(g)
        if isinstance(gi, int) and gi in DEMO_MAP:
            return DEMO_MAP[gi]
        gs = str(g)
        if gs in RENAME_MAP:
            return RENAME_MAP[gs].replace(" %", "")
        base = gs.replace("_pct", "").replace("_", " ").lower()
        mapping = {
            "white": "White",
            "black": "Black",
            "asian": "Asian",
            "female": "Female",
            "male": "Male",
            "age65": "Age 65+",
            "age 65": "Age 65+",
            "aian": "AIAN",
            "american indian alaska native": "AIAN",
        }
        return mapping.get(base, base.title())

    df["display_label"] = df[group_col].apply(prettify_label)

    # Sort by PDRR ascending (most under-represented first)
    df = df.sort_values("pdrr").reset_index(drop=True)

    x_labels = df["display_label"].tolist()

    # Parse enrollment and reference as proportions
    if value_col:
        enrollment = [_parse_pct_string(v) for v in df[value_col]]
    else:
        enrollment = [float(p) for p in df["pdrr"]]  # fallback

    if ref_col:
        reference = [_parse_pct_string(v) for v in df[ref_col]]
    else:
        reference = [1.0 / len(df)] * len(df)  # fallback uniform

    pdrr_vals = df["pdrr"].tolist()

    # Color enrollment bars: orange if under-represented, blue if at/above parity
    enroll_colors = [
        "#D55E00" if p < threshold else "#4C72B0"
        for p in pdrr_vals
    ]

    fig = go.Figure()

    # --- Bar 1: Enrollment % ---
    fig.add_bar(
        name="Predicted Enrollment",
        x=x_labels,
        y=[v * 100 for v in enrollment],
        marker_color=enroll_colors,
        text=[f"{v*100:.1f}%" for v in enrollment],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Predicted enrollment: %{y:.1f}%<extra></extra>"
        ),
    )

    # --- Bar 2: Population reference % ---
    fig.add_bar(
        name="Population Reference",
        x=x_labels,
        y=[v * 100 for v in reference],
        marker_color="rgba(100, 100, 100, 0.35)",
        marker_line=dict(color="rgba(100,100,100,0.6)", width=1),
        text=[f"{v*100:.1f}%" for v in reference],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Population reference: %{y:.1f}%<extra></extra>"
        ),
    )

    # --- PDRR annotation above each group ---
    max_y = max(max(v * 100 for v in enrollment), max(v * 100 for v in reference))
    annotation_y = max_y * 1.18

    for i, (label, pdrr) in enumerate(zip(x_labels, pdrr_vals)):
        color = "#D55E00" if pdrr < threshold else "#2a6496"
        fig.add_annotation(
            x=label,
            y=annotation_y,
            text=f"PDRR: {pdrr:.2f}",
            showarrow=False,
            font=dict(size=11, color=color, family="Arial"),
            bgcolor="rgba(255,255,255,0.7)",
            borderpad=2,
        )

    fig.update_layout(
        barmode="group",
        bargap=0.20,
        bargroupgap=0.05,
        title=dict(
            text="Enrollment vs. Population Reference by Demographic Group",
            font=dict(size=18),
            pad=dict(b=12),
        ),
        xaxis=dict(
            title="Demographic Group",
            type="category",
            tickangle=20,
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Percentage (%)",
            range=[0, annotation_y * 1.12],
            ticksuffix="%",
            tickfont=dict(size=12),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
        ),
        margin=dict(l=60, r=40, t=140, b=80),
        template="plotly_white",
        annotations=[
            dict(
                text=(
                    "Predicted enrollment compared to U.S. population reference. "
                    "PDRR = enrollment / reference (1.0 = parity). "
                    "Orange bars indicate under-representation (PDRR < 1.0)."
                ),
                xref="paper",
                yref="paper",
                x=0,
                y=1.13,
                showarrow=False,
                align="left",
                font=dict(size=12, color="#555"),
            ),
        ],
    )

    return fig
