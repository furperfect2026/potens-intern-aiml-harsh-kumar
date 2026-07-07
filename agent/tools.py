"""
Three callable tools for the triage agent.

Each tool is a real function that does actual work:
- lookup_policy: keyword search over policies.json
- search_similar_tickets: TF-IDF cosine similarity over knowledge_base.json
- draft_acknowledgment: template-based response generation

Design decision: Every tool returns a {"status": "ok"|"no_matches", ...} envelope.
The agent prompt explicitly handles no_matches as a signal to lower confidence.
This prevents silent failures where the agent pretends it found something useful.
"""

import json
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent.parent / "data"


def _load_json(filename: str) -> list[dict]:
    filepath = DATA_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def lookup_policy(category: str, issue_keywords: list[str]) -> dict:
    """
    Search company policies by category and keyword relevance.
    Returns matching policies sorted by keyword overlap.
    """
    policies = _load_json("policies.json")

    # Filter by category first, then rank by keyword matches in title+content
    category_matches = [p for p in policies if p["category"] == category.lower()]

    if not category_matches:
        # Fallback: search ALL policies by keywords if category doesn't match
        category_matches = policies

    scored = []
    keywords_lower = [kw.lower() for kw in issue_keywords]
    for policy in category_matches:
        searchable = (policy["title"] + " " + policy["content"]).lower()
        score = sum(1 for kw in keywords_lower if kw in searchable)
        if score > 0:
            scored.append((score, policy))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return {
            "status": "no_matches",
            "policies": [],
            "note": f"No policies matched category='{category}' with keywords={issue_keywords}. "
                    "This may indicate an unusual issue type. Consider lowering confidence."
        }

    # Return top 3 matches
    results = []
    for _, policy in scored[:3]:
        results.append({
            "id": policy["id"],
            "title": policy["title"],
            "content": policy["content"],
            "escalation_rule": policy["escalation_rule"],
            "sla_hours": policy["sla_hours"]
        })

    return {"status": "ok", "policies": results}


def search_similar_tickets(description: str, top_k: int = 3) -> dict:
    """
    Find past resolved tickets similar to the input using TF-IDF cosine similarity.
    
    Uses scikit-learn's TfidfVectorizer — simple but effective for short text.
    A production system would use embeddings, but TF-IDF is honest and interpretable
    for a 24-hour take-home.
    """
    tickets = _load_json("knowledge_base.json")

    if not description or not description.strip():
        return {
            "status": "no_matches",
            "tickets": [],
            "note": "Empty or blank search query provided. Cannot find similar tickets."
        }

    corpus = [t["description"] for t in tickets]
    corpus.append(description)  # query is the last element

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Similarity between the query (last doc) and all tickets
    query_vec = tfidf_matrix[-1]
    similarities = cosine_similarity(query_vec, tfidf_matrix[:-1]).flatten()

    # Pair tickets with their similarity scores and sort descending
    scored = list(zip(similarities, tickets))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Filter out very low similarity scores (below 0.05 is essentially noise)
    MIN_SIMILARITY = 0.05
    top_results = [(sim, t) for sim, t in scored[:top_k] if sim >= MIN_SIMILARITY]

    if not top_results:
        return {
            "status": "no_matches",
            "tickets": [],
            "note": "No similar past tickets found above the similarity threshold. "
                    "This may be a novel issue type. Consider lowering confidence."
        }

    results = []
    for sim, ticket in top_results:
        results.append({
            "id": ticket["id"],
            "description": ticket["description"][:200],  # truncate for readability
            "category": ticket["category"],
            "priority": ticket["priority"],
            "resolution": ticket["resolution"],
            "similarity_score": round(float(sim), 3)
        })

    return {"status": "ok", "tickets": results}


def draft_acknowledgment(category: str, priority: str, summary: str) -> dict:
    """
    Generate a professional customer-facing acknowledgment message.
    
    Uses templates rather than LLM generation — faster, deterministic,
    and doesn't consume API tokens for boilerplate text.
    """
    # SLA lookup for response time commitment
    sla_map = {
        "P0": "15 minutes",
        "P1": "4 hours",
        "P2": "1 business day"
    }

    team_map = {
        "billing": "billing team",
        "technical": "engineering team",
        "account": "account support team",
        "feature_request": "product team",
        "compliance": "compliance and legal team",
        "general": "support team"
    }

    sla = sla_map.get(priority, "1 business day")
    team = team_map.get(category, "support team")

    # Priority-specific opening
    if priority == "P0":
        urgency_line = ("We recognize this is a critical issue and have flagged it "
                        "for immediate attention.")
    elif priority == "P1":
        urgency_line = "We understand the urgency and have prioritized this accordingly."
    else:
        urgency_line = "We've received your message and will look into this for you."

    draft = (
        f"Hello,\n\n"
        f"Thank you for reaching out. We've received your support request "
        f"regarding: {summary}\n\n"
        f"{urgency_line}\n\n"
        f"Your ticket has been routed to our {team}. "
        f"You can expect an initial response within {sla}.\n\n"
        f"If you have any additional details that might help us resolve this "
        f"faster, please reply to this message.\n\n"
        f"Best regards,\n"
        f"Support Team"
    )

    return {
        "status": "ok",
        "draft": draft,
        "metadata": {
            "sla_commitment": sla,
            "assigned_team": team,
            "priority_level": priority
        }
    }


# Registry mapping tool names to functions — used by the agent loop
TOOL_REGISTRY = {
    "lookup_policy": lookup_policy,
    "search_similar_tickets": search_similar_tickets,
    "draft_acknowledgment": draft_acknowledgment,
}
