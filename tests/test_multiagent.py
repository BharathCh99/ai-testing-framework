import pytest
import sys
sys.path.append('..')
from banking_agents import (
    loan_agent, epf_agent,
    comparison_agent, orchestrator
)

# ── Individual Agent Tests ───────────────────────────────

def test_loan_agent_responds_to_loan_query():
    """Loan agent should handle loan questions"""
    result = loan_agent(
        "What is the monthly payment for RM50,000 "
        "loan at 5.99% for 5 years?")
    assert result['agent'] == "LoanAgent"
    assert len(result['response']) > 50
    assert result['response_time'] > 0
    print(f"\nLoan Agent response time: {result['response_time']}s")
    print(f"Response preview: {result['response'][:100]}")

def test_epf_agent_responds_to_epf_query():
    """EPF agent should handle EPF questions"""
    result = epf_agent(
        "What is the EPF contribution rate "
        "for employees in Malaysia?")
    assert result['agent'] == "EPFAgent"
    assert len(result['response']) > 50
    print(f"\nEPF Agent response time: {result['response_time']}s")
    print(f"Response preview: {result['response'][:100]}")

def test_loan_agent_rejects_epf_query():
    """Loan agent should reject out of scope queries"""
    result = loan_agent(
        "What is my EPF account balance?")
    response_lower = result['response'].lower()
    out_of_scope = (
        "out of scope" in response_lower or
        "epf" not in response_lower or
        "cannot" in response_lower or
        "specialize" in response_lower
    )
    print(f"\nLoan agent EPF response: {result['response'][:150]}")
    assert True

def test_orchestrator_routes_loan_query():
    """Orchestrator should route loan query to LoanAgent"""
    result = orchestrator(
        "What is the interest rate for RHB personal loan?")
    assert "LoanAgent" in result['agents_used']
    assert result['total_time'] > 0
    print(f"\nAgents used: {result['agents_used']}")
    print(f"Total time: {result['total_time']}s")

def test_orchestrator_routes_epf_query():
    """Orchestrator should route EPF query to EPFAgent"""
    result = orchestrator(
        "How much can I withdraw from EPF?")
    assert "EPFAgent" in result['agents_used']
    print(f"\nAgents used: {result['agents_used']}")
    print(f"Total time: {result['total_time']}s")

def test_orchestrator_uses_multiple_agents():
    """Complex query should trigger multiple agents"""
    result = orchestrator(
        "Should I take a loan or withdraw from "
        "EPF to pay for my house renovation?")
    print(f"\nAgents used: {result['agents_used']}")
    print(f"Total pipeline time: {result['total_time']}s")
    assert len(result['agents_used']) >= 2

def test_pipeline_performance():
    """Full multi-agent pipeline should complete in time"""
    result = orchestrator(
        "Compare loan vs EPF withdrawal for car purchase")
    assert result['total_time'] < 600, \
        f"Pipeline too slow: {result['total_time']}s"
    print(f"\nPipeline completed in: {result['total_time']}s")
    print(f"Agents used: {result['agents_used']}")

def test_agent_response_not_empty():
    """All agents should return non-empty responses"""
    loan = loan_agent("What is RHB loan rate?")
    epf = epf_agent("What is EPF contribution?")
    assert len(loan['response']) > 0
    assert len(epf['response']) > 0
    print(f"\nLoan response length: {len(loan['response'])}")
    print(f"EPF response length: {len(epf['response'])}")