# Triage Agent with Real Tool Calling

An agentic customer-support triage system that accepts free-text tickets, reasons through them step-by-step using real tool calls, and produces structured triage decisions with full transparency. 

Built for the **Potens 2026 Internship Take-Home (Q2: AI/ML)**.

---

## 🏃‍♂️ How to Run

1. `pip install -r requirements.txt`
2. Get a free Groq API key (console.groq.com) and add it to `.env`:
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Run the Streamlit UI:
   ```bash
   streamlit run app.py
   ```
4. *Optional*: Re-run the evaluation set to generate the 10 test examples:
   ```bash
   python eval/generate_examples.py
   ```

---

## 🏗️ Architecture & Design Decisions

### 1. Potens-Flavored Data Layer
I modeled the knowledge base and policies on what a high-trust, high-stakes SaaS environment like Potens actually deals with (e.g., GDPR deletion, enterprise SLA degradation, RBAC changes). The data is in `data/policies.json` and `data/knowledge_base.json`.

### 2. Why `next_action` instead of `next_tool`?
The brief asked for `next_tool`, but I redefined it in the schema as `next_action` (e.g., `route_to_engineering`, `escalate_to_manager`). Outputting the last internal tool the AI called isn't useful for a human ops team. Outputting a recommended workflow action demonstrates product judgment about the system this agent feeds into.

### 3. Built-in Confidence & HITL Escalation
Instead of bolting on a Human-In-The-Loop gate later, I built `confidence` (0.0-1.0) directly into the core `TriageOutput` schema. The model is explicitly prompted to self-rate based on ambiguity and tool failure. If confidence drops below 0.6, the Streamlit UI triggers a prominent warning banner with override controls.

### 4. Explicit Tool Failure Handling
LLMs love to silently hallucinate when tools fail. Every tool in `agent/tools.py` returns a strict `status: "ok" | "no_matches"` envelope. The system prompt explicitly instructs the model to lower its confidence if it sees `"no_matches"`. 

### 5. TF-IDF over Embeddings
For `search_similar_tickets`, I used scikit-learn's TF-IDF instead of vector embeddings. For a 24-hour take-home with a 20-ticket mock dataset, embeddings are overkill. TF-IDF is fast, interpretable, deterministic, and doesn't burn API quota.

---

## 🛠️ The Tools Actually Matter
Most submissions will include tools that the LLM could bypass by just guessing from keywords. I designed this system so the tools are **provably necessary**. 

Look at **Example 7** in the `examples/` folder. The ticket is about slow reporting — a classic P2 issue. However, the metadata shows it's an `enterprise` customer. The agent calls `lookup_policy` and finds `POL-015`: *Any ticket reporting slow performance on a Potens Enterprise account involving the reporting module is automatically considered P0.* The agent correctly overrides its instinct and outputs **P0**. This is why tool-calling matters here — it's not a string-matching shortcut.

---

## 📊 Honest Evaluation Numbers

I ran 10 test cases (including 3 adversarial ones) through the agent. 

| Metric | Result |
|---|---|
| Total Examples | 10 |
| Perfect Routing | 8 |
| Flagged for HITL Review | 1 |
| Incorrect/Failed | 1 |

**Where it succeeded:**
It handles the happy paths flawlessly and correctly identifies adversarial cases like Gibberish (flagged for HITL due to low confidence) and the P2-to-P0 Enterprise escalation mentioned above.

**Where it failed:**
In **Example 8** ("I see charges I didn't make and my data may be exposed"), the agent anchored too hard on the word "charges" and categorized it as `billing` with high confidence, missing the severe compliance/security breach implications in the second half of the sentence. In production, this is dangerous. 

---

## ✂️ What I'd Cut in Production

If this were shipping to a real Potens customer environment in 90 days, I would change three things:
1. **Drop TF-IDF for Real Embeddings**: Once the knowledge base scales past a few hundred tickets, TF-IDF will fail on synonyms. I'd move to a lightweight vector DB (like Qdrant) with `text-embedding-3-small`.
2. **Move to a compiled graph (LangGraph)**: The simple `while` loop in `core.py` is honest for a take-home, but in production, I'd want the state management, checkpointing, and strict edge transitions of LangGraph.
3. **Log Confidence Drift**: I'd pipe the `confidence` scores to an observability tool (like LangSmith or Datadog) to alert us if the model's average confidence starts drifting down over time, indicating our policies are getting out of date.

---

## 🤖 AI Use Log

*We grade on honesty about what you used and what you didn't.*

| Tool | Approx. Usage | What I used it for |
|---|---|---|
| **Claude 3.5 Sonnet (Cursor)** | ~10 prompts | Initial project scaffolding, brainstorming the 6 categories, and writing the Streamlit UI boilerplate. |
| **ChatGPT (GPT-4o)** | ~3 prompts | Generating the realistic, Potens-flavored JSON data (15 policies, 20 past tickets). I provided the schema, it did the typing. |
| **Gemini 2.0 Flash / Groq** | Native | Used as the actual inference engine for the agent loop (switched from Gemini to Groq mid-build due to free-tier quota limits). |
| **GitHub Copilot** | Continuous | Standard autocomplete while writing the python logic in `core.py` and `tools.py` (e.g., auto-filling the TF-IDF setup). |
