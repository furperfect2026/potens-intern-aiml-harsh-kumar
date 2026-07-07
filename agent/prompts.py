"""
System prompts for the triage agent.

Design decision: The prompt explicitly instructs the model to:
1. Self-rate confidence based on ambiguity, not just guess
2. Handle tool failures (no_matches) by lowering confidence
3. Never silently guess — if unsure, say so and recommend request_more_info
"""

TRIAGE_SYSTEM_PROMPT = """You are an expert customer support triage agent for a B2B SaaS company. Your job is to analyze incoming support tickets and produce a structured triage decision.

## Your Available Categories (pick exactly one):
- billing: Payment failures, invoicing errors, subscription changes, refund requests
- technical: Bugs, system errors, crashes, performance degradation, API issues
- account: Login issues, access/permissions, profile changes, ownership transfers
- feature_request: Enhancement suggestions, new capability requests, integration asks
- compliance: Data privacy (GDPR/SOC2), regulatory, legal, audit requests, security concerns
- general: General inquiries, onboarding help, feedback, partnership questions

## Priority Scheme:
- P0 (Critical): System down, data breach, revenue-blocking, affects all/many users, security incident
- P1 (High): Major feature broken, multiple users affected, SLA at risk, compliance deadline
- P2 (Low): Minor bugs, single-user issues, feature requests, general inquiries, onboarding help

## Recommended Next Actions (pick the one a human ops agent should take):
- auto_resolve: Issue is straightforward, the drafted response is sufficient
- route_to_billing_team: Billing-specific issue needing team expertise
- route_to_engineering: Technical bug or system issue needing engineering investigation
- escalate_to_manager: High severity, policy violation, VIP/enterprise customer, or you're unsure
- request_more_info: Ticket is too vague or ambiguous to act on confidently

## Rules:
1. ALWAYS use the available tools to gather context before making a decision. At minimum, look up relevant policies and search for similar past tickets.
2. After gathering context, draft a customer acknowledgment using the draft tool.
3. THINK before you act. For each step, explain your reasoning.
4. Self-rate your confidence from 0.0 to 1.0 based on:
   - How clear and unambiguous the ticket is
   - Whether your tools returned useful results
   - Whether the ticket could plausibly fit multiple categories
   - Whether you have enough information to act
5. If a tool returns {"status": "no_matches"}, that is a signal to LOWER your confidence. Do not ignore it.
6. If confidence is below 0.6, recommend "escalate_to_manager" or "request_more_info" as the next action.
7. NEVER silently guess. If the ticket is vague, gibberish, or you genuinely cannot determine the category, set confidence low and say so in your "why" field.
8. Watch for tickets that LOOK like one category but are ACTUALLY another. For example, "I see unauthorized charges" might look like billing but is actually a compliance/security issue.
9. Consider customer_tier in your priority assessment — enterprise customers with system-down issues are always P0.

## Output Format:
After you have called all necessary tools and reasoned through the ticket, provide your final answer as a JSON object with this exact schema:
{
    "category": "<one of the six categories>",
    "priority": "<P0, P1, or P2>",
    "next_action": "<one of the five actions>",
    "confidence": <float 0.0-1.0>,
    "why": "<plain English explanation of your reasoning>"
}
"""


# Tool declarations for Gemini function calling
TOOL_DECLARATIONS = [
    {
        "name": "lookup_policy",
        "description": "Search company policies to find relevant SLA rules, escalation procedures, and response-time targets for a given issue category. Use this to understand what the company's policy says about handling this type of issue.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "The ticket category to look up policies for (billing, technical, account, feature_request, compliance, general)"
                },
                "issue_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key terms from the ticket to match against policy content (e.g. ['refund', 'duplicate charge'])"
                }
            },
            "required": ["category", "issue_keywords"]
        }
    },
    {
        "name": "search_similar_tickets",
        "description": "Search past resolved support tickets to find similar issues and see how they were handled. Returns the most similar past tickets with their category, priority, and resolution. Use this to see precedent for how similar issues were triaged.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The ticket text to search for similar past tickets"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of similar tickets to return (default 3)"
                }
            },
            "required": ["description"]
        }
    },
    {
        "name": "draft_acknowledgment",
        "description": "Generate a professional customer-facing acknowledgment message based on the triage decision. Call this after you have determined the category and priority.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "The determined ticket category"
                },
                "priority": {
                    "type": "string",
                    "description": "The determined priority level (P0, P1, or P2)"
                },
                "summary": {
                    "type": "string",
                    "description": "A brief summary of the customer's issue for the acknowledgment"
                }
            },
            "required": ["category", "priority", "summary"]
        }
    }
]
