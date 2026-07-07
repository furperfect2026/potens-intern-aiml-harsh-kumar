"""
Streamlit UI for the Triage Agent.

Layout:
- Left: input panel (ticket text + metadata)
- Right: results panel (triage decision + reasoning trace + draft response)
- HITL banner when confidence < 0.6
"""

import streamlit as st
import json
import sys
import os

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.core import run_triage
from agent.models import TicketInput, TriageOutput


# --- Page config ---
st.set_page_config(
    page_title="Triage Agent",
    page_icon="🎯",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .priority-p0 {
        background-color: #dc2626;
        color: white;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 14px;
    }
    .priority-p1 {
        background-color: #f59e0b;
        color: white;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 14px;
    }
    .priority-p2 {
        background-color: #3b82f6;
        color: white;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 14px;
    }
    .category-badge {
        background-color: #1f2937;
        color: #e5e7eb;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 14px;
    }
    .confidence-high { color: #16a34a; font-weight: bold; }
    .confidence-mid { color: #f59e0b; font-weight: bold; }
    .confidence-low { color: #dc2626; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def render_priority_badge(priority: str) -> str:
    css_class = f"priority-{priority.lower()}"
    return f'<span class="{css_class}">{priority}</span>'


def render_confidence(confidence: float) -> str:
    if confidence >= 0.8:
        css = "confidence-high"
    elif confidence >= 0.6:
        css = "confidence-mid"
    else:
        css = "confidence-low"
    return f'<span class="{css}">{confidence:.0%}</span>'


def render_reasoning_trace(result: TriageOutput):
    """Show each reasoning step as expandable sections."""
    st.subheader("🔍 Reasoning Trace")

    for step in result.reasoning:
        if step.action == "think":
            with st.expander(f"💭 Step {step.step}: Thinking", expanded=False):
                st.write(step.detail)

        elif step.action == "call_tool":
            with st.expander(
                f"🔧 Step {step.step}: Called `{step.tool_name}`", expanded=False
            ):
                st.write(step.detail)
                if step.tool_input:
                    st.code(json.dumps(step.tool_input, indent=2), language="json")

        elif step.action == "observe":
            with st.expander(
                f"👁️ Step {step.step}: Observation from `{step.tool_name}`",
                expanded=False,
            ):
                st.write(step.detail)
                if step.tool_output:
                    # Truncate large outputs for readability
                    output_str = json.dumps(step.tool_output, indent=2)
                    if len(output_str) > 2000:
                        output_str = output_str[:2000] + "\n... (truncated)"
                    st.code(output_str, language="json")


def main():
    st.title("🎯 Triage Agent")
    st.caption(
        "AI-powered support ticket triage with real tool calling and full reasoning transparency"
    )

    st.divider()

    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("📝 Submit a Ticket")

        ticket_text = st.text_area(
            "Ticket content",
            height=200,
            placeholder="Describe the customer's issue here...",
        )

        col_tier, col_area = st.columns(2)
        with col_tier:
            customer_tier = st.selectbox(
                "Customer tier (optional)",
                options=["(not specified)", "free", "pro", "enterprise"],
            )
        with col_area:
            product_area = st.text_input(
                "Product area (optional)", placeholder="e.g. billing, dashboard, API"
            )

        triage_btn = st.button("🚀 Run Triage", type="primary", use_container_width=True)

    with col_output:
        if triage_btn and ticket_text.strip():
            tier = customer_tier if customer_tier != "(not specified)" else None
            area = product_area.strip() or None

            ticket = TicketInput(text=ticket_text, customer_tier=tier, product_area=area)

            with st.spinner("Agent is reasoning..."):
                try:
                    result = run_triage(ticket)
                    st.session_state["last_result"] = result
                except Exception as e:
                    st.error(f"Agent error: {e}")
                    st.stop()

        elif triage_btn and not ticket_text.strip():
            st.warning("Please enter some ticket text first.")

        # Render results if we have them
        if "last_result" in st.session_state:
            result = st.session_state["last_result"]

            # HITL banner
            if result.needs_human_review:
                st.warning(
                    "⚠️ **Low confidence — human review recommended.** "
                    "The agent is not confident enough to auto-triage this ticket. "
                    "Review the reasoning below and confirm or override.",
                    icon="🧑",
                )

                # Override controls
                with st.expander("✏️ Override triage decision", expanded=True):
                    override_cat = st.selectbox(
                        "Category",
                        options=["billing", "technical", "account",
                                 "feature_request", "compliance", "general"],
                        index=["billing", "technical", "account",
                               "feature_request", "compliance", "general"].index(
                            result.category
                        ),
                    )
                    override_pri = st.selectbox(
                        "Priority",
                        options=["P0", "P1", "P2"],
                        index=["P0", "P1", "P2"].index(result.priority),
                    )
                    if st.button("✅ Confirm Override"):
                        result.category = override_cat
                        result.priority = override_pri
                        result.needs_human_review = False
                        st.success("Triage decision overridden and confirmed.")
                        st.rerun()

            # Result card
            st.subheader("📋 Triage Decision")

            col_cat, col_pri, col_conf = st.columns(3)
            with col_cat:
                st.markdown(f"**Category**")
                st.markdown(
                    f'<span class="category-badge">{result.category}</span>',
                    unsafe_allow_html=True,
                )
            with col_pri:
                st.markdown(f"**Priority**")
                st.markdown(render_priority_badge(result.priority), unsafe_allow_html=True)
            with col_conf:
                st.markdown(f"**Confidence**")
                st.markdown(render_confidence(result.confidence), unsafe_allow_html=True)

            st.markdown(f"**Next Action:** `{result.next_action}`")

            # Why explanation
            st.info(f"**Why:** {result.why}")

            # Draft response
            if result.draft_response:
                with st.expander("✉️ Draft Customer Response", expanded=False):
                    st.text(result.draft_response)

            st.divider()

            # Reasoning trace
            render_reasoning_trace(result)

            # Raw JSON output
            with st.expander("📦 Raw JSON Output", expanded=False):
                st.code(result.model_dump_json(indent=2), language="json")


if __name__ == "__main__":
    main()
