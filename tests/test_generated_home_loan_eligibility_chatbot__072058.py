Here are the 5 pytest test functions:

```
import pytest
import ollama
from deepeval.test_case import LLMTestCase
from deepeeval.metrics import AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM


class OllamaModel:
    def __init__(self):
        pass

    def predict(self, input_text):
        # Your prediction logic goes here
        return "You are eligible for a home loan"


@pytest.mark.parametrize("input_text", [
    "I am 30 and earn S$60k",
    "I have no income, I'm unemployed",
    "My age is 25, and I earn S$80k"
])
def test_home_loan_eligibility(input_text):
    model = OllamaModel()
    prediction = model.predict(input_text)
    assert "eligible for a home loan" in prediction


@pytest.mark.parametrize("input_text", [
    "I am 18 and earn S$20k",
    "My age is 45, but I have no income",
    "I'm retired, my age is 65"
])
def test_home_loan_ineligibility(input_text):
    model = OllamaModel()
    prediction = model.predict(input_text)
    assert "eligible for a home loan" not in prediction


@pytest.mark.parametrize("input_text", [
    "What are the requirements to apply?",
    "How do I increase my credit score?",
    "Can I get a loan with no income?"
])
def test_home_loan_chatbot_info(input_text):
    model = OllamaModel()
    prediction = model.predict(input_text)
    assert "You are eligible for a home loan" not in prediction


@pytest.mark.parametrize("input_text", [
    "I have 20% down payment, how much can I borrow?",
    "What is the interest rate for this loan?",
    "How long does it take to process my application?"
])
def test_home_loan_chatbot_answers(input_text):
    model = OllamaModel()
    prediction = model.predict(input_text)
    assert "You are eligible for a home loan" not in prediction


@pytest.mark.parametrize("input_text", [
    "What is the minimum income required?",
    "Can I get a loan with no employment history?",
    "How do I increase my credit score?"
])
def test_home_loan_chatbot_boundary_cases(input_text):
    model = OllamaModel()
    prediction = model.predict(input_text)
    assert "You are eligible for a home loan" not in prediction
```