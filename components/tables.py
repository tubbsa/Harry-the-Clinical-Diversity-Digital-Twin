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

# Columns to drop from display — PDRR_raw is redundant now cap is removed
COLUMNS_TO_DROP = {"PDRR_raw", "pdrr_raw", "PDRR Raw"}


def _coerce_int_like(x: object) -> object:
    """Convert int-like values (e.g., '3', 3.0) to int; otherwise return as-is."""
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
    """Apply DEMO_MAP and RENAME_MAP to likely label columns (display only)."""
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
    """
    Format a column of proportions (0–1) as percentage strings.
    Values already > 1 are treated as already-scaled percentages.
    """
    num = pd.to_numeric(series, errors="coerce")
    # Scale up if looks like proportion
    prop_mask = num.notna() & (num >= 0) & (num <= 1)
    scaled = num.copy()
    scaled[prop_mask] = num[prop_mask] * 100
    return scaled.map(lambda x: "" if pd.isna(x) else f"{x:.1f}%")


def _format_percent_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Display-only formatting:
    - Enrollment (value) and Population Reference columns → formatted as xx.x%
    - PDRR and Score columns → formatted as decimals
    """
    out = df.copy()

    percent_labels = set(RENAME_MAP.values())

    # --- Format Enrollment column ---
    val_col = "value" if "value" in out.columns else (
              "Value" if "Value" in out.columns else None)
    comp_col = "component" if "component" in out.columns else (
               "Component" if "Component" in out.columns else None)

    if val_col is not None:
        if comp_col is not None:
            percent_mask = out[comp_col].isin(percent_labels)
        else:
            percent_mask = pd.Series(True, index=out.index)

        val_num = pd.to_numeric(out[val_col], errors="coerce")
        prop_mask = val_num.notna() & (val_num >= 0) & (val_num <= 1)
        mult_mask = percent_mask & prop_mask
        if mult_mask.any():
            out.loc[mult_mask, val_col] = (val_num.loc[mult_mask] * 100).astype(float)
            val_num = pd.to_numeric(out[val_col], errors="coerce")

        if percent_mask.any():
            pct_num = pd.to_numeric(out.loc[percent_mask, val_col], errors="coerce")
            out.loc[percent_mask, val_col] = pct_num.map(
                lambda x: "" if pd.isna(x) else f"{x:.1f}%")

        non_pct_mask = ~percent_mask
        if non_pct_mask.any():
            nonpct_num = pd.to_numeric(out.loc[non_pct_mask, val_col], errors="coerce")
            out.loc[non_pct_mask, val_col] = nonpct_num.map(
                lambda x: "" if pd.isna(x) else f"{x:.3f}")

    # --- Format Population Reference column as percentage ---
    for ref_col in ("population_reference", "disease_prevalence",
                    "Population Reference", "Disease Prevalence"):
        if ref_col in out.columns:
            out[ref_col] = _format_percent_col(out[ref_col])
            break

    return out


def render_breakdown_table(breakdown: Optional[pd.DataFrame]) -> None:
    """
    Renders an ICER breakdown table safely.
    Never renders "nothing" — always shows a placeholder if empty.
    """
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

    # -----------------------------
    # Display-only transformations
    # -----------------------------
    display_df = breakdown.copy()

    # Drop redundant columns before renaming
    cols_to_drop = [c for c in display_df.columns if c in COLUMNS_TO_DROP]
    if cols_to_drop:
        display_df = display_df.drop(columns=cols_to_drop)

    display_df = _apply_display_labels(display_df)

    # Rename columns to friendly headers
    display_df = display_df.rename(
        columns={c: HEADER_MAP.get(c, c) for c in display_df.columns}
    )

    # Drop again by friendly name in case renaming revealed duplicates
    cols_to_drop_friendly = [c for c in display_df.columns if c in COLUMNS_TO_DROP]
    if cols_to_drop_friendly:
        display_df = display_df.drop(columns=cols_to_drop_friendly)

    display_df = _format_percent_values(display_df)

    st.dataframe(display_df, use_container_width=True)

    # Download the display table (friendly labels)
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📊 Download ICER Breakdown as CSV",
        data=csv_bytes,
        file_name="icer_breakdown.csv",
        mime="text/csv",
    )

