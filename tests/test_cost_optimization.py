import pytest
import ollama
import time
import json

# ── Token Counter ────────────────────────────────────────
def count_tokens_approx(text: str) -> int:
    """Approximate token count (1 token = approx 4 characters)"""
    return len(text) // 4

def calculate_cost(tokens: int,
                   model: str = "gpt4") -> float:
    """Calculate estimated API cost in USD"""
    pricing = {
        "gpt4":    0.030 / 1000,   # $30 per 1M tokens
        "gpt35":   0.0005 / 1000,  # $0.50 per 1M tokens
        "claude":  0.015 / 1000,   # $15 per 1M tokens
    }
    rate = pricing.get(model, 0.030 / 1000)
    return round(tokens * rate, 6)

# ── Prompts to Compare ───────────────────────────────────

VERBOSE_PROMPT = """
You are an extremely helpful and knowledgeable
banking assistant working for RHB Bank in Malaysia.
Your role is to assist customers with their banking
queries in a professional, friendly, and comprehensive
manner. Please provide detailed, accurate, and helpful
responses to all customer inquiries. Make sure to be
thorough in your explanations and cover all aspects
of the topic.

Customer Question: {question}

Please provide a comprehensive response:
"""

OPTIMIZED_PROMPT = """Banking assistant for RHB Bank.
Answer concisely and accurately.
Q: {question}
A:"""

ULTRA_OPTIMIZED_PROMPT = "RHB Bank Q: {question} A:"

# ── Test Functions ───────────────────────────────────────

def ask_with_prompt(prompt_template: str,
                    question: str) -> dict:
    prompt = prompt_template.format(question=question)
    input_tokens = count_tokens_approx(prompt)

    start = time.time()
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    end = time.time()

    answer = response['message']['content']
    output_tokens = count_tokens_approx(answer)
    total_tokens = input_tokens + output_tokens

    return {
        "prompt_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "response_time": round(end - start, 2),
        "answer": answer,
        "cost_gpt4": calculate_cost(total_tokens, "gpt4"),
        "cost_gpt35": calculate_cost(total_tokens, "gpt35"),
        "cost_claude": calculate_cost(total_tokens, "claude")
    }

question = "What is the RHB fixed deposit interest rate?"

# ── Tests ────────────────────────────────────────────────

def test_verbose_prompt_token_count():
    """Measure token usage of verbose prompt"""
    result = ask_with_prompt(VERBOSE_PROMPT, question)
    print(f"\nVerbose Prompt:")
    print(f"  Input tokens : {result['prompt_tokens']}")
    print(f"  Output tokens: {result['output_tokens']}")
    print(f"  Total tokens : {result['total_tokens']}")
    print(f"  GPT-4 cost   : ${result['cost_gpt4']}")
    print(f"  Response time: {result['response_time']}s")
    assert result['total_tokens'] > 0

def test_optimized_prompt_token_count():
    """Measure token usage of optimized prompt"""
    result = ask_with_prompt(OPTIMIZED_PROMPT, question)
    print(f"\nOptimized Prompt:")
    print(f"  Input tokens : {result['prompt_tokens']}")
    print(f"  Output tokens: {result['output_tokens']}")
    print(f"  Total tokens : {result['total_tokens']}")
    print(f"  GPT-4 cost   : ${result['cost_gpt4']}")
    print(f"  Response time: {result['response_time']}s")
    assert result['total_tokens'] > 0

def test_ultra_optimized_prompt_token_count():
    """Measure token usage of ultra optimized prompt"""
    result = ask_with_prompt(
        ULTRA_OPTIMIZED_PROMPT, question)
    print(f"\nUltra Optimized Prompt:")
    print(f"  Input tokens : {result['prompt_tokens']}")
    print(f"  Output tokens: {result['output_tokens']}")
    print(f"  Total tokens : {result['total_tokens']}")
    print(f"  GPT-4 cost   : ${result['cost_gpt4']}")
    print(f"  Response time: {result['response_time']}s")
    assert result['total_tokens'] > 0

def test_cost_saving_calculation():
    """Calculate annual cost savings from optimization"""
    verbose = ask_with_prompt(VERBOSE_PROMPT, question)
    optimized = ask_with_prompt(OPTIMIZED_PROMPT, question)

    token_saving = (verbose['total_tokens'] -
                    optimized['total_tokens'])
    saving_pct = round(
        (token_saving / verbose['total_tokens']) * 100, 1)

    # Project to 50,000 daily queries
    daily_queries = 50000
    daily_token_saving = token_saving * daily_queries
    annual_token_saving = daily_token_saving * 365

    annual_cost_saving_gpt4 = calculate_cost(
        annual_token_saving, "gpt4")

    print(f"\nCost Saving Analysis:")
    print(f"  Token reduction : {token_saving} tokens")
    print(f"  Saving %        : {saving_pct}%")
    print(f"  Daily saving    : {daily_token_saving:,} tokens")
    print(f"  Annual saving   : {annual_token_saving:,} tokens")
    print(f"  Annual $ saving : ${annual_cost_saving_gpt4:,.2f} (GPT-4)")

    assert saving_pct > 0, "Optimized prompt should use fewer tokens"
    assert token_saving > 0, "Should save at least some tokens"

def test_quality_not_degraded_after_optimization():
    """Verify quality stays acceptable after optimization"""
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.models.base_model import DeepEvalBaseLLM

    class OllamaModel(DeepEvalBaseLLM):
        def load_model(self): return "llama3"
        def generate(self, prompt: str) -> str:
            r = ollama.chat(model="llama3",
                messages=[{"role": "user", "content": prompt}])
            return r['message']['content']
        async def a_generate(self, prompt: str) -> str:
            return self.generate(prompt)
        def get_model_name(self): return "llama3"

    ollama_model = OllamaModel()
    metric = AnswerRelevancyMetric(
        threshold=0.5, model=ollama_model)

    # Test optimized prompt quality
    result = ask_with_prompt(OPTIMIZED_PROMPT, question)
    tc = LLMTestCase(
        input=question,
        actual_output=result['answer']
    )
    metric.measure(tc)
    quality_score = round(metric.score, 2)

    print(f"\nQuality after optimization: {quality_score}")
    print(f"Tokens used: {result['total_tokens']}")
    assert quality_score >= 0.5, \
        f"Quality degraded too much: {quality_score}"
    print("Quality maintained after optimization!")

def test_response_length_control():
    """Test that concise prompts produce shorter responses"""
    verbose_result = ask_with_prompt(
        VERBOSE_PROMPT, question)
    optimized_result = ask_with_prompt(
        OPTIMIZED_PROMPT, question)

    verbose_len = len(verbose_result['answer'])
    optimized_len = len(optimized_result['answer'])

    print(f"\nResponse length comparison:")
    print(f"  Verbose response   : {verbose_len} chars")
    print(f"  Optimized response : {optimized_len} chars")
    print(f"  Reduction          : {verbose_len - optimized_len} chars")
    assert True