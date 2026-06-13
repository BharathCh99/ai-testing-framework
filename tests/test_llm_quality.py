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

knowledge_base = {
    "fixed deposit": "RHB Fixed Deposit offers interest rates from 2.85% to 3.10% per annum.",
    "personal loan": "RHB Personal Loan offers amounts from RM5,000 to RM150,000 at 5.99% per annum.",
    "savings account": "RHB Savings Account requires minimum balance of RM20 at 0.25% per annum."
}

def retrieve_context(question: str) -> str:
    for key, value in knowledge_base.items():
        if key in question.lower():
            return value
    return "No relevant information found."

def ask_rag(question: str) -> dict:
    context = retrieve_context(question)
    prompt = f"""Use only the information below to answer.
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

# ── Test Cases ───────────────────────────────────────────

def test_fixed_deposit_relevancy():
    result = ask_rag("What is the interest rate for fixed deposit?")
    test_case = LLMTestCase(
        input=result['question'],
        actual_output=result['answer'],
        retrieval_context=[result['context']]
    )
    metric = AnswerRelevancyMetric(threshold=0.5, model=ollama_model)
    metric.measure(test_case)
    assert metric.score >= 0.5, f"Relevancy too low: {metric.score}"

def test_fixed_deposit_faithfulness():
    result = ask_rag("What is the interest rate for fixed deposit?")
    test_case = LLMTestCase(
        input=result['question'],
        actual_output=result['answer'],
        retrieval_context=[result['context']]
    )
    metric = FaithfulnessMetric(threshold=0.5, model=ollama_model)
    metric.measure(test_case)
    assert metric.score >= 0.5, f"Faithfulness too low: {metric.score}"

def test_personal_loan_relevancy():
    result = ask_rag("How much can I borrow for a personal loan?")
    test_case = LLMTestCase(
        input=result['question'],
        actual_output=result['answer'],
        retrieval_context=[result['context']]
    )
    metric = AnswerRelevancyMetric(threshold=0.5, model=ollama_model)
    metric.measure(test_case)
    assert metric.score >= 0.5, f"Relevancy too low: {metric.score}"

def test_response_time_under_threshold():
    result = ask_rag("What is the minimum balance for savings account?")
    assert result['response_time'] < 300, \
        f"Response too slow: {result['response_time']}s"