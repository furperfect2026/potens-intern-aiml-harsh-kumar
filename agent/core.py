"""
Core agent loop using Gemini 2.0 Flash with native function calling.

The loop:
1. Send ticket + system prompt to Gemini with tool declarations
2. If model returns tool calls → execute them via TOOL_REGISTRY, feed results back
3. Repeat until model produces a final text response (no more tool calls)
4. Parse the structured triage output from the response
5. Check confidence → flag for human review if < 0.6

Design decision: Using google-genai SDK (not the older google-generativeai)
for cleaner function calling support. The agent loop is intentionally simple —
no framework (LangChain, CrewAI) because the assignment tests whether we can
wire LLM components ourselves, not whether we can install a framework.
"""

import json
import os
import re
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from agent.models import TicketInput, TriageOutput, ReasoningStep
from agent.prompts import TRIAGE_SYSTEM_PROMPT, TOOL_DECLARATIONS
from agent.tools import TOOL_REGISTRY

load_dotenv()


def _build_tools() -> list[types.Tool]:
    """Convert our tool declarations into Gemini Tool objects."""
    function_declarations = []
    for decl in TOOL_DECLARATIONS:
        function_declarations.append(
            types.FunctionDeclaration(
                name=decl["name"],
                description=decl["description"],
                parameters=decl["parameters"]
            )
        )
    return [types.Tool(function_declarations=function_declarations)]


def _parse_triage_output(text: str, reasoning_trace: list[ReasoningStep]) -> TriageOutput:
    """
    Extract the JSON triage decision from the model's final text response.
    Handles cases where the model wraps JSON in markdown code fences.
    """
    # Try to find JSON in the response (model sometimes wraps in ```json ... ```)
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        raw_json = json_match.group(1)
    else:
        # Try to find raw JSON object
        json_match = re.search(r'\{[^{}]*"category"[^{}]*\}', text, re.DOTALL)
        if json_match:
            raw_json = json_match.group(0)
        else:
            # Fallback: the entire text might be JSON
            raw_json = text.strip()

    parsed = json.loads(raw_json)

    # Build the output, injecting the reasoning trace we captured
    output = TriageOutput(
        category=parsed["category"],
        priority=parsed["priority"],
        next_action=parsed["next_action"],
        confidence=float(parsed["confidence"]),
        reasoning=reasoning_trace,
        why=parsed["why"],
        draft_response=parsed.get("draft_response"),
        needs_human_review=float(parsed["confidence"]) < 0.6
    )

    return output


def _execute_tool_call(function_call) -> dict:
    """Execute a tool call and return the result. Handles unknown tools gracefully."""
    tool_name = function_call.name
    tool_args = dict(function_call.args) if function_call.args else {}

    if tool_name not in TOOL_REGISTRY:
        return {
            "status": "error",
            "message": f"Unknown tool: {tool_name}. Available tools: {list(TOOL_REGISTRY.keys())}"
        }

    tool_fn = TOOL_REGISTRY[tool_name]

    try:
        result = tool_fn(**tool_args)
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": f"Tool '{tool_name}' failed with error: {str(e)}"
        }


def run_triage(ticket: TicketInput) -> TriageOutput:
    """
    Main entry point. Takes a ticket, runs the agent loop, returns a triage decision.
    
    The loop continues until the model stops requesting tool calls
    (max 10 iterations as a safety limit to prevent infinite loops).
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Build the initial user message with ticket details
    user_message = f"Triage this support ticket:\n\n{ticket.text}"
    if ticket.customer_tier:
        user_message += f"\n\nCustomer tier: {ticket.customer_tier}"
    if ticket.product_area:
        user_message += f"\n\nProduct area: {ticket.product_area}"

    tools = _build_tools()
    reasoning_trace: list[ReasoningStep] = []
    step_counter = 0

    # Initial request
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_message)])]

    MAX_ITERATIONS = 10

    for iteration in range(MAX_ITERATIONS):
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=TRIAGE_SYSTEM_PROMPT,
                tools=tools,
                temperature=0.2,  # low temperature for consistent triage decisions
            )
        )

        candidate = response.candidates[0]
        parts = candidate.content.parts

        # Check if the model wants to call tools
        tool_call_parts = [p for p in parts if p.function_call]
        text_parts = [p for p in parts if p.text]

        if not tool_call_parts:
            # No tool calls — model is done reasoning, parse final output
            final_text = "".join(p.text for p in text_parts)

            # Log the final thinking step
            step_counter += 1
            reasoning_trace.append(ReasoningStep(
                step=step_counter,
                action="think",
                detail=f"Final decision: {final_text[:300]}"
            ))

            # Add assistant response to contents for completeness
            contents.append(candidate.content)

            return _parse_triage_output(final_text, reasoning_trace)

        # Process each tool call
        # First, add the assistant's response (with tool calls) to contents
        contents.append(candidate.content)

        # Log any thinking text that came with the tool calls
        if text_parts:
            step_counter += 1
            thinking_text = "".join(p.text for p in text_parts)
            reasoning_trace.append(ReasoningStep(
                step=step_counter,
                action="think",
                detail=thinking_text[:500]
            ))

        # Execute each tool call and collect results
        function_response_parts = []
        for part in tool_call_parts:
            fc = part.function_call
            step_counter += 1

            # Log the tool call
            tool_args = dict(fc.args) if fc.args else {}
            reasoning_trace.append(ReasoningStep(
                step=step_counter,
                action="call_tool",
                detail=f"Calling {fc.name}",
                tool_name=fc.name,
                tool_input=tool_args
            ))

            # Execute
            result = _execute_tool_call(fc)

            # Log the observation
            step_counter += 1
            reasoning_trace.append(ReasoningStep(
                step=step_counter,
                action="observe",
                detail=f"{fc.name} returned status={result.get('status', 'unknown')}",
                tool_name=fc.name,
                tool_output=result
            ))

            # Build the function response part
            function_response_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response=result
                )
            )

        # Add tool results back to the conversation
        contents.append(types.Content(role="user", parts=function_response_parts))

    # Safety: if we hit max iterations, return a low-confidence fallback
    return TriageOutput(
        category="general",
        priority="P2",
        next_action="escalate_to_manager",
        confidence=0.1,
        reasoning=reasoning_trace,
        why="Agent reached maximum iteration limit without producing a final decision. "
            "Escalating to manager for manual review.",
        needs_human_review=True
    )
