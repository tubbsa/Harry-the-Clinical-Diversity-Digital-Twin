# components/tables.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st

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

HEADER_MAP = {
    "component":            "Component",
    "value":                "Enrollment",
    "disease_prevalence":   "Population Reference",   # backward compat
    "population_reference": "Population Reference",   # new column name
    "PDRR_raw":             "PDRR Raw",
    "pdrr_raw":             "PDRR Raw",
    "PDRR":                 "PDRR",
    "pdrr":                 "PDRR",
    "score":                "Score",
    "domain":               "Domain",
    "group":                "Group",
}

# Columns to drop from display - PDRR_raw is redundant now cap is removed
COLUMNS_TO_DROP = {"PDRR_raw", "pdrr_raw", "PDRR Raw"}


def _coerce_int_like(x: object) -> object:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if s.isdigit():
            try:
                return int(s)
            except Exception:
                return x
    return x


def _apply_display_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    label_cols = ["group", "component", "demographic", "category", "label", "subgroup"]
    for col in label_cols:
        if col not in out.columns:
            continue
        s = out[col]
        coerced = s.map(_coerce_int_like)
        mapped = coerced.map(DEMO_MAP)
        out[col] = mapped.fillna(s)
    if "component" in out.columns:
        out["component"] = out["component"].replace(RENAME_MAP)
    return out


def _format_percent_col(series: pd.Series) -> pd.Series:
    """Format a column of proportions (0-1) as percentage strings."""
    num = pd.to_numeric(series, errors="coerce")
    prop_mask = num.notna() & (num >= 0) & (num <= 1)
    scaled = num.copy()
    scaled[prop_mask] = num[prop_mask] * 100
    return scaled.map(lambda x: "" if pd.isna(x) else f"{x:.1f}%")


def _format_percent_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Display-only formatting. Called AFTER column renaming.
    Checks for both pre-rename and post-rename column names.
    - Enrollment -> formatted as xx.x%
    - Population Reference -> formatted as xx.x%
    - PDRR and Score -> left as decimals
    """
    out = df.copy()

    percent_labels = set(RENAME_MAP.values())

    # Find enrollment column - check post-rename name first
    val_col = next(
        (c for c in ("Enrollment", "value", "Value") if c in out.columns),
        None
    )

    # Find component column - check post-rename name first
    comp_col = next(
        (c for c in ("Component", "component") if c in out.columns),
        None
    )

    # --- Format Enrollment column ---
    if val_col is not None:
        if comp_col is not None:
            percent_mask = out[comp_col].isin(percent_labels)
        else:
            percent_mask = pd.Series(True, index=out.index)

        if percent_mask.any():
            out.loc[percent_mask, val_col] = _format_percent_col(
                out.loc[percent_mask, val_col]
            )

        non_pct_mask = ~percent_mask
        if non_pct_mask.any():
            nonpct_num = pd.to_numeric(out.loc[non_pct_mask, val_col], errors="coerce")
            out.loc[non_pct_mask, val_col] = nonpct_num.map(
                lambda x: "" if pd.isna(x) else f"{x:.3f}"
            )

    # --- Format Population Reference column as percentage ---
    for ref_col in ("Population Reference", "population_reference",
                    "Disease Prevalence", "disease_prevalence"):
        if ref_col in out.columns:
            out[ref_col] = _format_percent_col(out[ref_col])
            break

    return out


def render_breakdown_table(breakdown: Optional[pd.DataFrame]) -> None:
    """Renders an ICER breakdown table safely."""
    st.subheader("ICER Breakdown Table")

    if breakdown is None:
        st.info("No breakdown table available yet.")
        return
    if not isinstance(breakdown, pd.DataFrame):
        st.warning(f"Expected a pandas DataFrame, got {type(breakdown)}.")
        st.write(breakdown)
        return
    if breakdown.empty:
        st.info("Breakdown table is empty.")
        return

    display_df = breakdown.copy()

    # Drop redundant columns before renaming
    cols_to_drop = [c for c in display_df.columns if c in COLUMNS_TO_DROP]
    if cols_to_drop:
        display_df = display_df.drop(columns=cols_to_drop)

    # Apply demographic label display names
    display_df = _apply_display_labels(display_df)

    # Rename columns to friendly headers
    display_df = display_df.rename(
        columns={c: HEADER_MAP.get(c, c) for c in display_df.columns}
    )

    # Drop redundant columns again by friendly name
    cols_to_drop_friendly = [c for c in display_df.columns if c in COLUMNS_TO_DROP]
    if cols_to_drop_friendly:
        display_df = display_df.drop(columns=cols_to_drop_friendly)

    # Format percentage columns - runs AFTER renaming
    display_df = _format_percent_values(display_df)

    st.dataframe(display_df, use_container_width=True)

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📊 Download ICER Breakdown as CSV",
        data=csv_bytes,
        file_name="icer_breakdown.csv",
        mime="text/csv",
    )
