"""
Pydantic models for the triage agent's input/output schema.

Design decision: next_action is a recommended action for a human ops agent
(e.g. "route_to_engineering"), NOT the last tool the AI called internally.
This makes the output actionable in a real ops workflow.
"""

from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, Field


class TicketInput(BaseModel):
    """Incoming support ticket with optional context."""
    text: str = Field(..., description="Free-text ticket content from the customer")
    customer_tier: Optional[Literal["free", "pro", "enterprise"]] = Field(
        default=None,
        description="Customer's subscription tier, if known"
    )
    product_area: Optional[str] = Field(
        default=None,
        description="Product area the ticket relates to, if known"
    )


class ReasoningStep(BaseModel):
    """One step in the agent's reasoning trace."""
    step: int
    action: Literal["think", "call_tool", "observe"]
    detail: str
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[dict] = None


# The six categories the agent can assign
CATEGORIES = Literal[
    "billing",
    "technical",
    "account",
    "feature_request",
    "compliance",
    "general"
]

# Priority levels
PRIORITIES = Literal["P0", "P1", "P2"]

# Recommended next actions for the human ops team
NEXT_ACTIONS = Literal[
    "auto_resolve",
    "route_to_billing_team",
    "route_to_engineering",
    "escalate_to_manager",
    "request_more_info"
]


class TriageOutput(BaseModel):
    """
    Structured triage decision.
    
    confidence is self-reported by the model (0.0-1.0) based on
    ticket clarity, category overlap, and available evidence.
    When confidence < 0.6, needs_human_review is set to True.
    """
    category: CATEGORIES
    priority: PRIORITIES
    next_action: NEXT_ACTIONS
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Model's self-assessed confidence in this triage decision"
    )
    reasoning: list[ReasoningStep] = Field(default_factory=list)
    why: str = Field(
        ...,
        description="Plain-English explanation of why this triage decision was made"
    )
    draft_response: Optional[str] = Field(
        default=None,
        description="Customer-facing acknowledgment draft, if generated"
    )
    needs_human_review: bool = Field(
        default=False,
        description="True when confidence < 0.6 — flags for human-in-the-loop review"
    )
