import ollama
import time

# ── Agent 1: Loan Calculator Agent ──────────────────────
def loan_agent(query: str) -> dict:
    """Specialized agent for loan calculations"""
    system = """You are a loan specialist agent.
You ONLY answer questions about loans, interest rates,
monthly payments and loan eligibility.
If asked about anything else, say 'OUT OF SCOPE'.
Always include numbers in your response."""

    start = time.time()
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query}
        ]
    )
    end = time.time()
    return {
        "agent": "LoanAgent",
        "response": response['message']['content'],
        "response_time": round(end - start, 2)
    }

# ── Agent 2: EPF Specialist Agent ───────────────────────
def epf_agent(query: str) -> dict:
    """Specialized agent for EPF queries"""
    system = """You are an EPF (Employee Provident Fund)
specialist agent for Malaysia.
You ONLY answer questions about EPF contributions,
withdrawals, retirement planning and EPF accounts.
If asked about anything else, say 'OUT OF SCOPE'.
Always include specific EPF rules and percentages."""

    start = time.time()
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query}
        ]
    )
    end = time.time()
    return {
        "agent": "EPFAgent",
        "response": response['message']['content'],
        "response_time": round(end - start, 2)
    }

# ── Agent 3: Financial Comparison Agent ─────────────────
def comparison_agent(option_a: str,
                     option_b: str,
                     question: str) -> dict:
    """Specialized agent for financial comparisons"""
    system = """You are a financial comparison agent.
You receive two financial options and recommend
the best one based on cost, risk and benefit.
Always give a clear RECOMMENDATION at the end."""

    query = f"""Compare these two options for: {question}

Option A (Loan):
{option_a}

Option B (EPF):
{option_b}

Which is better and why?"""

    start = time.time()
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query}
        ]
    )
    end = time.time()
    return {
        "agent": "ComparisonAgent",
        "response": response['message']['content'],
        "response_time": round(end - start, 2)
    }

# ── Orchestrator ─────────────────────────────────────────
def orchestrator(user_query: str) -> dict:
    """Routes query to right agents and combines results"""
    print(f"\nOrchestrator received: {user_query[:50]}...")

    query_lower = user_query.lower()
    pipeline_start = time.time()
    agents_used = []
    results = {}

    # Route to appropriate agents
    if any(word in query_lower for word in
           ["loan", "borrow", "interest", "repay"]):
        print("  Routing to LoanAgent...")
        results['loan'] = loan_agent(user_query)
        agents_used.append("LoanAgent")

    if any(word in query_lower for word in
           ["epf", "kwsp", "retirement", "withdraw"]):
        print("  Routing to EPFAgent...")
        results['epf'] = epf_agent(user_query)
        agents_used.append("EPFAgent")

    # If both agents used, run comparison
    if 'loan' in results and 'epf' in results:
        print("  Routing to ComparisonAgent...")
        results['comparison'] = comparison_agent(
            results['loan']['response'],
            results['epf']['response'],
            user_query
        )
        agents_used.append("ComparisonAgent")

    pipeline_end = time.time()

    return {
        "query": user_query,
        "agents_used": agents_used,
        "results": results,
        "total_time": round(
            pipeline_end - pipeline_start, 2),
        "final_answer": results.get(
            'comparison', {}).get('response') or
            list(results.values())[0]['response']
            if results else "No agent could handle this"
    }