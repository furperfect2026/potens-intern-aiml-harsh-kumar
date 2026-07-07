# Potens Triage Agent (Q2: AI/ML Take-Home)

Hey! Thanks for reviewing my submission. This is my take on the AI-powered support ticket triage system. 

Instead of just hooking up an LLM to a basic prompt, I focused on building a system that actually reasons through a ticket using real tools, and provides a transparent output that a human ops team could actually use.

---

## How to Run

1. `pip install -r requirements.txt`
2. Get a free Groq API key (console.groq.com) and add it to `.env`:
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Run the Streamlit UI:
   ```bash
   streamlit run app.py
   ```
4. *Optional*: Re-run the evaluation script to see the agent handle the 10 test examples:
   ```bash
   python eval/generate_examples.py
   ```

---

## How I Approached This

### 1. Making the tools actually matter
I noticed a lot of LLMs can just guess the category from keywords (e.g., seeing the word "credit card" and guessing "billing"). I wanted to prove my tools were actually doing the work. 

If you look at **Example 7** in the `examples/` folder, the ticket complains about slow reporting. Normally, that's a `P2` bug. But because the customer is an `enterprise` user, the agent uses the `lookup_policy` tool, finds the Potens SLA policy for enterprise reporting (`POL-015`), and overrides its initial instinct to correctly label it a **P0**. It's not just string-matching; it's actually following policy.

### 2. Built for humans (`next_action` vs `next_tool`)
The prompt mentioned tracking the `next_tool`, but I realized that knowing the last API the LLM called isn't very helpful for a support agent looking at the dashboard. I redefined the schema to output a `next_action` (like `route_to_engineering` or `escalate_to_manager`). I think this makes the agent a lot more useful in a real workflow.

### 3. Built-in Confidence & Human-in-the-loop
The system explicitly asks the model to rate its own confidence (0.0 to 1.0). If a tool fails (returns `"no_matches"`) or the ticket is just confusing, the confidence drops. Anything below 0.6 triggers a warning in the UI, letting a human agent override the decision.

### 4. TF-IDF over Vector Embeddings
For finding similar past tickets, I used scikit-learn's TF-IDF. Vector embeddings are cool, but for a 24-hour challenge with a small mock dataset, TF-IDF is faster, deterministic, and doesn't eat up API quota. 

---

## Honest Evaluation Numbers

I ran 10 test cases through the agent, including a few adversarial ones to try and break it. 

| Metric | Result |
|---|---|
| Total Examples | 10 |
| Handled Perfectly | 8 |
| Flagged for Human Review | 1 |
| Incorrect | 1 |

**Where it succeeded:**
It handles standard tickets flawlessly and correctly caught the P2-to-P0 Enterprise escalation (Example 7). It also successfully flagged gibberish (Example 9) for human review due to low confidence.

**Where it failed:**
In **Example 8** ("I see charges I didn't make and my data may be exposed"), the agent anchored too hard on the word "charges" and categorized it as `billing`. It totally missed the severe compliance/security breach implications in the second half of the sentence. Definitely a gap I'd want to fix before launching.

---

## What I'd Change For Production

If I were actually shipping this to a Potens customer environment in 90 days, I'd upgrade a few things:
1. **Swap TF-IDF for Real Embeddings**: Once the knowledge base grows, TF-IDF will struggle with synonyms. I'd move to a lightweight vector DB with `text-embedding-3-small`.
2. **Move to a state graph (like LangGraph)**: The simple `while` loop in `core.py` works great for a prototype, but in production, I'd want the strict state management and checkpointing that a framework like LangGraph provides.
3. **Telemetry on Confidence**: I'd pipe the `confidence` scores to Datadog or LangSmith. If average confidence starts drifting down over time, it's a good alert that our policies might be outdated.

---

## AI Use Log

As requested, here is my honest log of how I used AI to help build this over the last 24 hours:

| Tool | Approx. Usage | What I used it for |
|---|---|---|
| **Claude 3.5 Sonnet** | ~4-5 prompts | Bouncing around initial architecture ideas (like using TF-IDF vs embeddings) and asking for suggestions on good edge-case scenarios to test against. |
| **ChatGPT (GPT-4o)** | ~2 prompts | I wrote the JSON schema, but I used ChatGPT to quickly generate the realistic Potens-flavored mock data (the 15 policies and 20 past tickets) because writing those out by hand is tedious. |
| **GitHub Copilot** | Continuous | Standard IDE autocomplete while writing the core Python logic in `core.py` and `tools.py`. |
| **Groq (Llama-3.3)** | Native | Used purely as the inference engine for the agent loop (I initially tried Gemini but hit the free-tier rate limits, so I swapped to Groq for testing). |
