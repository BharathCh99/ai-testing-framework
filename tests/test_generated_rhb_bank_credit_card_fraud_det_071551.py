Here are the 5 pytest test functions:

```
import pytest
import ollama
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM

class OllamaModel:
    def __init__(self, model_name):
        self.model = ollama.load(model_name)

    def predict(self, prompt):
        return self.model(prompt)

class TestRHBBankCreditCardFraudDetectionChatbot(LLMTestCase):

    @pytest.mark.parametrize("prompt, expected", [
        ("Transaction amount exceeds $5000", True),
        ("Payment method is not Visa/Mastercard", False),
        ("Last 3 digits of card are 123", True)
    ])
    def test_fraud_detection(self, prompt, expected):
        model = OllamaModel("RHB-Bank-Credit-Card-Fraud-Detection")
        result = self.model.predict(prompt)
        assert self.get_answer_relevance(result, prompt) == AnswerRelevancyMetric.HIGH if expected else AnswerRelevancyMetric.LOW

    def test_edge_case(self):
        model = OllamaModel("RHB-Bank-Credit-Card-Fraud-Detection")
        result = self.model.predict("Transaction amount is $0")
        assert self.get_answer_relevance(result, "Transaction amount is $0") == AnswerRelevancyMetric.LOW

    def test_boundary_case(self):
        model = OllamaModel("RHB-Bank-Credit-Card-Fraud-Detection")
        result = self.model.predict("Last 4 digits of card are 0000")
        assert self.get_answer_relevance(result, "Last 4 digits of card are 0000") == AnswerRelevancyMetric.MEDIUM

    def test_invalid_input(self):
        model = OllamaModel("RHB-Bank-Credit-Card-Fraud-Detection")
        result = self.model.predict("Invalid input: ABC123")
        assert self.get_answer_relevance(result, "Invalid input: ABC123") == AnswerRelevancyMetric.LOW

    def test_default_case(self):
        model = OllamaModel("RHB-Bank-Credit-Card-Fraud-Detection")
        result = self.model.predict("Default case: no fraud detected")
        assert self.get_answer_relevance(result, "Default case: no fraud detected") == AnswerRelevancyMetric.HIGH
```