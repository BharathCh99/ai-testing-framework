import pytest
import ollama
import time
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM

class OllamaModel(DeepEvalBaseLLM):
    def load_model(self): return "llama3"
    def generate(self, prompt: str) -> str:
        r = ollama.chat(model="llama3",
            messages=[{"role":"user","content":prompt}])
        return r['message']['content']
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
    def get_model_name(self): return "llama3"

# ── Expanded Banking Knowledge Base ─────────────────────
knowledge_base = {
    "fixed deposit": "RHB Fixed Deposit offers interest rates from 2.85% to 3.10% per annum depending on tenure from 1 to 12 months. Minimum deposit is RM1,000.",
    "personal loan": "RHB Personal Loan offers amounts from RM5,000 to RM150,000 with tenure up to 5 years at interest rates from 5.99% per annum. Applicant must be 21-60 years old.",
    "savings account": "RHB Savings Account requires a minimum balance of RM20 and offers interest rate of 0.25% per annum. No monthly fees.",
    "epf withdrawal": "EPF Account 2 allows withdrawal for housing, education, health, and hajj purposes. Members must be Malaysian citizens or permanent residents.",
    "epf contribution": "EPF contribution rate for employees is 11% of monthly salary. Employers contribute 13% for salaries below RM5,000 and 12% for salaries above RM5,000.",
    "epf retirement": "EPF members can make full withdrawal at age 55 or opt for monthly pension payments. The basic savings target is RM240,000 at age 55.",
    "credit card": "RHB Credit Card offers cashback up to 2% on all purchases, travel miles, and zero interest for 20 days. Annual fee is RM150, waived for first year.",
    "home loan": "RHB Home Loan offers financing up to 90% of property value with tenure up to 35 years. Interest rates start from 3.5% per annum based on BLR.",
    "fraud": "If you suspect fraud on your RHB account, immediately call RHB Contact Centre at 03-9206-8118, available 24/7. Freeze your card via RHB Mobile Banking app.",
    "kyc": "KYC (Know Your Customer) requires valid IC or passport, proof of address, and income documents. All RHB accounts require KYC compliance under Bank Negara regulations."
}

def retrieve_context(question: str) -> str:
    question_lower = question.lower()
    for key, value in knowledge_base.items():
        if key in question_lower:
            return value
    return "No relevant information found."

def ask_rag(question: str) -> dict:
    context = retrieve_context(question)
    prompt = f"""You are a banking assistant. Use only the 
information below to answer the question accurately.
Information: {context}
Question: {question}
Answer:"""
    start = time.time()
    response = ollama.chat(model="llama3",
        messages=[{"role":"user","content":prompt}])
    end = time.time()
    return {
        "question": question,
        "context": context,
        "answer": response['message']['content'],
        "response_time": round(end - start, 2)
    }

ollama_model = OllamaModel()

def make_metrics():
    return (
        AnswerRelevancyMetric(threshold=0.5, model=ollama_model),
        FaithfulnessMetric(threshold=0.5, model=ollama_model)
    )

# ── RHB Tests ────────────────────────────────────────────

def test_rhb_fixed_deposit():
    result = ask_rag("What is the minimum deposit for RHB fixed deposit?")
    relevancy, faithfulness = make_metrics()
    tc = LLMTestCase(input=result['question'],
        actual_output=result['answer'],
        retrieval_context=[result['context']])
    relevancy.measure(tc)
    assert relevancy.score >= 0.5, f"Score: {relevancy.score}"

def test_rhb_personal_loan_age():
    result = ask_rag("What is the age requirement for RHB personal loan?")
    relevancy, faithfulness = make_metrics()
    tc = LLMTestCase(input=result['question'],
        actual_output=result['answer'],
        retrieval_context=[result['context']])
    relevancy.measure(tc)
    assert relevancy.score >= 0.5, f"Score: {relevancy.score}"

def test_rhb_home_loan():
    result = ask_rag("How much home loan can I get from RHB?")
    relevancy, faithfulness = make_metrics()
    tc = LLMTestCase(input=result['question'],
        actual_output=result['answer'],
        retrieval_context=[result['context']])
    relevancy.measure(tc)
    assert relevancy.score >= 0.5, f"Score: {relevancy.score}"

def test_rhb_credit_card():
    result = ask_rag("What cashback does RHB credit card offer?")
    relevancy, faithfulness = make_metrics()
    tc = LLMTestCase(input=result['question'],
        actual_output=result['answer'],
        retrieval_context=[result['context']])
    relevancy.measure(tc)
    assert relevancy.score >= 0.5, f"Score: {relevancy.score}"

def test_rhb_fraud_response():
    result = ask_rag("What should I do if I suspect fraud on my RHB account?")
    relevancy, faithfulness = make_metrics()
    tc = LLMTestCase(input=result['question'],
        actual_output=result['answer'],
        retrieval_context=[result['context']])
    faithfulness.measure(tc)
    assert faithfulness.score >= 0.5, f"Score: {faithfulness.score}"

# ── EPF Tests ────────────────────────────────────────────

def test_epf_contribution_rate():
    result = ask_rag("What is the EPF contribution rate for employees?")
    relevancy, faithfulness = make_metrics()
    tc = LLMTestCase(input=result['question'],
        actual_output=result['answer'],
        retrieval_context=[result['context']])
    relevancy.measure(tc)
    assert relevancy.score >= 0.5, f"Score: {relevancy.score}"

def test_epf_withdrawal_purpose():
    result = ask_rag("What are the valid purposes for EPF withdrawal?")
    relevancy, faithfulness = make_metrics()
    tc = LLMTestCase(input=result['question'],
        actual_output=result['answer'],
        retrieval_context=[result['context']])
    relevancy.measure(tc)
    assert relevancy.score >= 0.5, f"Score: {relevancy.score}"

def test_epf_retirement_age():
    result = ask_rag("At what age can I withdraw my EPF savings?")
    relevancy, faithfulness = make_metrics()
    tc = LLMTestCase(input=result['question'],
        actual_output=result['answer'],
        retrieval_context=[result['context']])
    relevancy.measure(tc)
    assert relevancy.score >= 0.5, f"Score: {relevancy.score}"

# ── Compliance Tests ─────────────────────────────────────

def test_kyc_requirements():
    result = ask_rag("What documents are required for KYC?")
    relevancy, faithfulness = make_metrics()
    tc = LLMTestCase(input=result['question'],
        actual_output=result['answer'],
        retrieval_context=[result['context']])
    faithfulness.measure(tc)
    assert faithfulness.score >= 0.5, f"Score: {faithfulness.score}"

# ── Performance Test ─────────────────────────────────────

def test_all_responses_under_300s():
    questions = [
        "What is EPF contribution rate?",
        "How do I report RHB fraud?",
        "What is RHB home loan rate?"
    ]
    for q in questions:
        result = ask_rag(q)
        assert result['response_time'] < 300, \
            f"Too slow for: {q} — {result['response_time']}s"