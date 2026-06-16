Here are the 5 pytest test functions as per the requirement:

```python
import pytest
import ollama
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM

class OllamaModel(DeepEvalBaseLLM):
    def __init__(self, model_name):
        super(OllamaModel, self).__init__()
        self.model_name = model_name

def test_epf_calculator_llm():
    @pytest.mark.parametrize("input_text", [
        "What is the maximum EPF contribution?",
        "Can I withdraw my EPF at 55?",
        "How much EPF can I contribute this year?"
    ])
    def test_input(input_text):
        model = OllamaModel("epf_calculator")
        response = ollama.generate(model, input_text)
        assert "EPF" in response
        assert "retirement" in response

def test_epf_calculator_edge_cases():
    @pytest.mark.parametrize("input_text", [
        "What is the maximum EPF contribution for a 30-year-old?",
        "Can I withdraw my EPF at 55 with interest?"
    ])
    def test_input(input_text):
        model = OllamaModel("epf_calculator")
        response = ollama.generate(model, input_text)
        assert "EPF" in response
        assert "retirement" in response

def test_epf_calculator_invalid_queries():
    @pytest.mark.parametrize("input_text", [
        "What is the meaning of life?",
        "Can I withdraw my EPF at 20?"
    ])
    def test_input(input_text):
        model = OllamaModel("epf_calculator")
        response = ollama.generate(model, input_text)
        assert "EPF" not in response
        assert "retirement" not in response

def test_epf_calculator_limit_queries():
    @pytest.mark.parametrize("input_text", [
        "What is the maximum EPF contribution limit?",
        "Can I contribute more than 20% to my EPF?"
    ])
    def test_input(input_text):
        model = OllamaModel("epf_calculator")
        response = ollama.generate(model, input_text)
        assert "EPF" in response
        assert "limit" in response

def test_epf_calculator_multiple_queries():
    input_texts = [
        "What is the maximum EPF contribution?",
        "Can I withdraw my EPF at 55?",
        "How much EPF can I contribute this year?"
    ]
    model = OllamaModel("epf_calculator")
    responses = [ollama.generate(model, text) for text in input_texts]
    assert all("EPF" in response and "retirement" in response for response in responses)
```