import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.core import run_triage
from agent.models import TicketInput

EXAMPLES = [
    {
        "id": "example_01_billing",
        "description": "Clear billing issue (happy path)",
        "input": {
            "text": "My credit card was charged twice for this month's Potens subscription. I need a refund for the duplicate $99 charge.",
            "customer_tier": "pro",
            "product_area": "billing"
        }
    },
    {
        "id": "example_02_technical",
        "description": "Technical bug (happy path)",
        "input": {
            "text": "The data export is failing with a 500 error every time I try to download the weekly report in CSV format.",
            "customer_tier": "pro",
            "product_area": "reporting"
        }
    },
    {
        "id": "example_03_account",
        "description": "Account access (happy path)",
        "input": {
            "text": "I can't log in. I keep getting 'invalid credentials' but I just reset my password 10 minutes ago.",
            "customer_tier": "free",
            "product_area": "auth"
        }
    },
    {
        "id": "example_04_feature",
        "description": "Feature request (low priority)",
        "input": {
            "text": "It would be great if the Potens dashboard had a dark mode. Working at night is blinding.",
            "customer_tier": "pro",
            "product_area": "ui"
        }
    },
    {
        "id": "example_05_enterprise_p0",
        "description": "Enterprise customer urgent outage",
        "input": {
            "text": "The entire Potens API is returning 502 Bad Gateway for all our production services. We are completely blocked.",
            "customer_tier": "enterprise",
            "product_area": "api"
        }
    },
    {
        "id": "example_06_general",
        "description": "Polite general inquiry",
        "input": {
            "text": "Hi team, I'm new to Potens. Is there a quick start guide or video tutorial on how to set up my first project?",
            "customer_tier": "free",
            "product_area": "onboarding"
        }
    },
    {
        "id": "example_07_tool_necessity",
        "description": "Looks P2, upgraded to P0 by tool (Enterprise reporting degradation)",
        "input": {
            "text": "The reporting module is loading very slowly. Takes almost 2 minutes to generate a chart. It works, just slowly.",
            "customer_tier": "enterprise",
            "product_area": "reporting"
        }
    },
    {
        "id": "example_08_adversarial_compliance",
        "description": "Billing language but actually a compliance/breach issue",
        "input": {
            "text": "I see charges on my account I didn't make. Also, when I logged in, I saw someone else's email and project data on my dashboard. I think my account was hacked and my data is exposed.",
            "customer_tier": "pro",
            "product_area": "billing"
        }
    },
    {
        "id": "example_09_adversarial_gibberish",
        "description": "Gibberish / Empty input (tests graceful failure)",
        "input": {
            "text": "asdf jkl; qwer uiuiuiuiui",
            "customer_tier": None,
            "product_area": None
        }
    },
    {
        "id": "example_10_adversarial_no_matches",
        "description": "Obscure edge case (tests no_matches tool path)",
        "input": {
            "text": "We need to integrate Potens with our on-prem AS/400 mainframe running OS/400 V4R5 using a custom SOAP endpoint. Is there a specific WSDL we should use for this legacy connection?",
            "customer_tier": "enterprise",
            "product_area": "integration"
        }
    }
]

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

def generate():
    EXAMPLES_DIR.mkdir(exist_ok=True)
    
    for ex in EXAMPLES:
        print(f"Running {ex['id']}...")
        ticket = TicketInput(**ex["input"])
        
        try:
            output = run_triage(ticket)
            
            result = {
                "id": ex["id"],
                "description": ex["description"],
                "input": ex["input"],
                "agent_output": output.model_dump()
            }
            
            out_file = EXAMPLES_DIR / f"{ex['id']}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
                
            print(f"  -> Saved to {out_file.name}")
        except Exception as e:
            print(f"  -> ERROR: {e}")

if __name__ == "__main__":
    generate()
