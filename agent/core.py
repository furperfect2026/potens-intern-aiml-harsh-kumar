"""
Core agent loop using Groq with native function calling.

The loop:
1. Send ticket + system prompt to Groq with tool declarations
2. If model returns tool calls → execute them via TOOL_REGISTRY, feed results back
3. Repeat until model produces a final text response (no more tool calls)
4. Parse the structured triage output from the response
5. Check confidence → flag for human review if < 0.6
"""

import json
import os
import re
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

from agent.models import TicketInput, TriageOutput, ReasoningStep
from agent.prompts import TRIAGE_SYSTEM_PROMPT, TOOL_DECLARATIONS
from agent.tools import TOOL_REGISTRY

load_dotenv()


def _build_tools() -> list[dict]:
    """Convert our tool declarations into Groq/OpenAI JSON schema Tool objects."""
    tools = []
    for decl in TOOL_DECLARATIONS:
        tools.append({
            "type": "function",
            "function": {
                "name": decl["name"],
                "description": decl["description"],
                "parameters": decl["parameters"]
            }
        })
    return tools


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


def run_triage(ticket: TicketInput) -> TriageOutput:
    """
    Main entry point. Takes a ticket, runs the agent loop, returns a triage decision.
    
    The loop continues until the model stops requesting tool calls
    (max 10 iterations as a safety limit to prevent infinite loops).
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Build the initial user message with ticket details
    user_message = f"Triage this support ticket:\n\n{ticket.text}"
    if ticket.customer_tier:
        user_message += f"\n\nCustomer tier: {ticket.customer_tier}"
    if ticket.product_area:
        user_message += f"\n\nProduct area: {ticket.product_area}"

    tools = _build_tools()
    reasoning_trace: list[ReasoningStep] = []
    step_counter = 0

    messages = [
        {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    MAX_ITERATIONS = 10

    for iteration in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            temperature=0.2,
            tool_choice="auto"
        )

        message = response.choices[0].message
        
        # Check if the model wants to call tools
        if not message.tool_calls:
            # No tool calls — model is done reasoning, parse final output
            final_text = message.content or ""
            
            step_counter += 1
            reasoning_trace.append(ReasoningStep(
                step=step_counter,
                action="think",
                detail=f"Final decision output generated."
            ))
            
            return _parse_triage_output(final_text, reasoning_trace)
            
        # Add assistant's response to messages history
        messages.append(message)
        
        # Log any thinking text that came alongside the tool call
        if message.content:
            step_counter += 1
            reasoning_trace.append(ReasoningStep(
                step=step_counter,
                action="think",
                detail=message.content[:500]
            ))
            
        # Process each tool call
        for tool_call in message.tool_calls:
            step_counter += 1
            
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments
            try:
                tool_args = json.loads(tool_args_str)
            except json.JSONDecodeError:
                tool_args = {}
                
            reasoning_trace.append(ReasoningStep(
                step=step_counter,
                action="call_tool",
                detail=f"Calling {tool_name}",
                tool_name=tool_name,
                tool_input=tool_args
            ))
            
            # Execute tool
            if tool_name not in TOOL_REGISTRY:
                result = {"status": "error", "message": f"Unknown tool: {tool_name}"}
            else:
                try:
                    result = TOOL_REGISTRY[tool_name](**tool_args)
                except Exception as e:
                    result = {"status": "error", "message": str(e)}
                    
            step_counter += 1
            reasoning_trace.append(ReasoningStep(
                step=step_counter,
                action="observe",
                detail=f"{tool_name} returned status={result.get('status', 'unknown')}",
                tool_name=tool_name,
                tool_output=result
            ))
            
            # Feed result back into conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(result)
            })

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

