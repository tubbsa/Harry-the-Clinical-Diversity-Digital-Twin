# ============================================================
# Main.py — Harry the Clinical Trial Diversity Digital Twin
# Corrected Version — Payload Dict Fix
# ============================================================

# ------------------------------------------------------------
# PATH FIX — Make parent folder importable
# ------------------------------------------------------------
import sys, os
import streamlit as st
import pandas as pd
import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))      # digital_twin/app
PROJECT_ROOT = os.path.dirname(APP_DIR)                   # digital_twin

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ------------------------------------------------------------
# IMPORTS (MATCH YOUR CURRENT FOLDERS)
# ------------------------------------------------------------
# ML + Scoring
from src.predictor import predict_proportions
from src.scoring import compute_icer_score
from src.narrative import build_narrative
from src.nfrules import recommend_nf
from src.bandit import bandit_optimize
from src.scoring_constants import DISEASE_PREVALENCE
from src.scoring_constants import DISEASE_PREVALENCE, SEX_BURDEN_MORTALITY
from src.clinical_reporter import build_llm_report




# Components
from components.form_inputs import render_form_and_collect_inputs
from components.score_tiles import render_score_tiles
from components.tables import render_breakdown_table

# Charts
from charts.rep_prev_diverging import make_rep_prev_diverging
from charts.pdrr_chart import make_pdrr_bar_chart

# Utils
from utils.constants import DISPLAY_LABELS, CATEGORY_ORDER
from utils.payload_builder import build_payload
from utils.gap_analysis import compute_largest_gaps

# PDF
from pdf.scorecard_pdf import generate_pdf_scorecard

# ------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Clinical Trial Diversity Digital Twin",
    layout="wide",
)

# Academic styling
st.markdown("""
<style>
html, body, [class*="css"]  { font-family: 'Georgia', serif !important; }
h1, h2, h3, h4 { font-family: 'Georgia', serif !important; color: #1A1A1A; }
.metric-good  { background-color: #D8F5D2; padding: 12px; border-radius: 6px; border-left: 6px solid #2E8B57; }
.metric-fair  { background-color: #FFF4CC; padding: 12px; border-radius: 6px; border-left: 6px solid #C9A227; }
.metric-poor  { background-color: #FADBD8; padding: 12px; border-radius: 6px; border-left: 6px solid #B03A2E; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------
st.title("Harry the Clinical Trial Diversity Digital Twin")
st.caption("A modular ICER scoring and clinical-trial diversity prediction platform.")

page = st.sidebar.radio("Navigation", ["Main", "Report"])

# ============================================================
# MAIN PAGE
# ============================================================
if page == "Main":

    inputs = render_form_and_collect_inputs()

    if inputs is not None:

        # -------------------------
        # Build payload
        # -------------------------
        payload = build_payload(inputs)

        # -------------------------
        # ML prediction
        # -------------------------
        result = predict_proportions(payload)
        preds = result["preds"] or {}

        st.subheader("Predicted Representation (%)")
        preds_pct = {k: (v * 100 if v is not None else np.nan) for k, v in preds.items()}
        preds_ordered = {k: preds_pct.get(k, np.nan) for k in CATEGORY_ORDER}

        df_preds = (
            pd.DataFrame([preds_ordered])
              .rename(columns=DISPLAY_LABELS)
              .round(1)
        )
        st.dataframe(df_preds, use_container_width=True)

        # -------------------------
        # ICER scoring
        # -------------------------
        total_score, breakdown = compute_icer_score(preds, payload)
        total_score_sex_adj, breakdown_sex_adj = compute_icer_score(
            preds, payload, burden_override=SEX_BURDEN_MORTALITY
        )

        render_score_tiles(total_score, breakdown)

        # -------------------------
        # Charts + tables
        # -------------------------
        rep_prev_fig = make_rep_prev_diverging(preds, DISEASE_PREVALENCE)
        st.plotly_chart(rep_prev_fig, use_container_width=True)

        pdrr_fig = make_pdrr_bar_chart(breakdown)
        st.plotly_chart(pdrr_fig, use_container_width=True)

        render_breakdown_table(breakdown)

        # -------------------------
        # Recommendations
        # -------------------------
        preds_with_score = preds.copy()
        preds_with_score["icer_score"] = (total_score / 21.0) * 100.0

        nf_actions = recommend_nf(payload, preds_with_score)
        bandit_actions = bandit_optimize(payload, preds_with_score)

        st.subheader("Recommended Actions")
        st.write(nf_actions + bandit_actions)

        # -------------------------
        # Report payload
        # -------------------------
        report_payload = {
            "trial_context": {
                "phase": payload.get("phase"),
                "conditions": payload.get("conditions"),
                "intervention": payload.get("intervention"),
            },
            "predicted_representation": preds,
            "icer_score": total_score,
            "recommended_actions": {
                "neuro_fuzzy": nf_actions,
                "bandit": bandit_actions,
            },
            "disclaimer": (
                "This report is informational only and does not constitute "
                "clinical guidance or medical advice."
            ),
        }

        # -------------------------
        # Clinical LLM narrative
        # -------------------------
        narrative = build_llm_report(report_payload)

        # -------------------------
        # Persist for Report page
        # -------------------------
        st.session_state["report_ready"] = True
        st.session_state["report"] = narrative
        st.session_state["nf_actions"] = nf_actions
        st.session_state["bandit_actions"] = bandit_actions
        st.session_state["breakdown"] = breakdown
        st.session_state["rep_prev_fig"] = rep_prev_fig
        st.session_state["pdrr_fig"] = pdrr_fig


# ============================================================
# REPORT PAGE
# ============================================================
if page == "Report":

    if not st.session_state.get("report_ready", False):
        st.info("Run the model on the Main page to generate a report.")
        st.stop()

    # Pull the full report dict (recommended). Fall back gracefully if older state keys exist.
    report = st.session_state.get("report", None)

    # Back-compat: if you still have st.session_state["narrative"] holding the full dict or a string
    if report is None:
        report = st.session_state.get("narrative", None)

    # Extract summary text safely
    if isinstance(report, dict):
        summary_text = report.get("summary") or report.get("narrative") or report.get("text") or ""
    else:
        summary_text = str(report) if report is not None else ""

    nf_actions = st.session_state.get("nf_actions", [])
    bandit_actions = st.session_state.get("bandit_actions", [])
    breakdown = st.session_state.get("breakdown", None)
    rep_prev_fig = st.session_state.get("rep_prev_fig", None)
    pdrr_fig = st.session_state.get("pdrr_fig", None)

    st.header("Clinical Trial Diversity Report")

    st.subheader("Narrative Summary")
    if summary_text.strip():
        st.write(summary_text)
    else:
        st.warning("No narrative summary found. Go back to **Main** and generate the report again.")

    st.subheader("Recommended Actions")
    combined_actions = []
    if isinstance(nf_actions, list):
        combined_actions += nf_actions
    else:
        combined_actions.append(nf_actions)

    if isinstance(bandit_actions, list):
        combined_actions += bandit_actions
    else:
        combined_actions.append(bandit_actions)

    if combined_actions:
        st.write(combined_actions)
    else:
        st.info("No recommended actions available.")

    st.subheader("Download ICER Scorecard (PDF)")

    # Guard: disable PDF generation if we don't have a report dict
    can_make_pdf = isinstance(report, dict) and bool(report)

    if st.button("📄 Generate PDF Scorecard", disabled=not can_make_pdf):
        try:
            pdf_bytes = generate_pdf_scorecard(
                report=report,  # ✅ KEY FIX: pass the report dict
                title="ICER Diversity Prediction Scorecard for Your Clinical Trial",
                breakdown=breakdown,
                rep_prev_fig=rep_prev_fig,
                pdrr_fig=pdrr_fig,
            )

            st.download_button(
                "⬇️ Download PDF Scorecard",
                data=pdf_bytes,
                file_name="icer_scorecard.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error("Failed to generate PDF.")
            st.exception(e)

    if not can_make_pdf:
        st.info("PDF generation is disabled until a report is generated on the **Main** page.")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.write("---")
st.write("""
**Clinical Trial Diversity Digital Twin**  
ICER Scoring • Representation Modeling • PDF Scorecards  
Designed for equitable clinical research.
""")
