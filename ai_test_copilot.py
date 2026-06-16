import ollama
import os
from datetime import datetime

# ── AI Test Copilot ──────────────────────────────────────
COPILOT_PROMPT = """You are an expert AI Test Engineer 
specializing in banking systems and LLM testing.

When given a testing requirement, generate exactly 5 
pytest test functions. Follow these rules:

1. Each test must be a complete pytest function
2. Use this exact import block at the top:
   import pytest
   import ollama
   from deepeval.test_case import LLMTestCase
   from deepeval.metrics import AnswerRelevancyMetric
   from deepeval.models.base_model import DeepEvalBaseLLM

3. Include OllamaModel class definition
4. Each test must call ollama and assert quality
5. Focus on banking domain relevance
6. Include edge cases and boundary conditions
7. Return ONLY valid Python code, no explanations

Generate tests for: {requirement}
"""

def generate_tests(requirement: str) -> str:
    """Use AI to generate test cases"""
    print(f"\nGenerating tests for: {requirement}")
    print("AI is thinking...")

    response = ollama.chat(
        model="llama3",
        messages=[{
            "role": "user",
            "content": COPILOT_PROMPT.format(
                requirement=requirement)
        }]
    )
    return response['message']['content']

def clean_code(raw: str) -> str:
    """Extract clean Python code from AI response"""
    lines = raw.split('\n')
    clean_lines = []
    in_code = False

    for line in lines:
        if 'python' in line:
            in_code = True
            continue
        elif '' in line and in_code:
            in_code = False
            continue
        elif in_code:
            clean_lines.append(line)

    if clean_lines:
        return '\n'.join(clean_lines)

    # If no code blocks found, return as-is
    return raw

def save_tests(code: str, requirement: str) -> str:
    """Save generated tests to file"""
    # Clean filename
    clean_req = requirement.lower()
    clean_req = clean_req.replace(' ', '_')
    clean_req = ''.join(
        c for c in clean_req if c.isalnum() or c == '_')
    clean_req = clean_req[:30]

    timestamp = datetime.now().strftime("%H%M%S")
    filename = f"test_generated_{clean_req}_{timestamp}.py"
    filepath = os.path.join("tests", filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)

    return filepath

def run_copilot():
    """Main AI Test Copilot interface"""
    print("=" * 60)
    print("  AI TEST COPILOT - Powered by llama3")
    print("=" * 60)
    print("\nI will generate pytest test cases automatically!")
    print("Type 'quit' to exit\n")

    requirements = [
        "RHB bank credit card fraud detection chatbot",
        "EPF retirement calculator AI assistant",
        "Home loan eligibility chatbot for Singapore bank"
    ]

    for req in requirements:
        print("\n" + "=" * 60)
        print(f"Requirement: {req}")
        print("=" * 60)

        # Generate tests
        raw_code = generate_tests(req)
        clean = clean_code(raw_code)

        # Save to file
        filepath = save_tests(clean, req)
        print(f"\nTests saved to: {filepath}")
        print("\nGenerated code preview:")
        print("-" * 40)
        # Show first 10 lines
        preview_lines = clean.split('\n')[:10]
        for line in preview_lines:
            print(line)
        print("...")
        print(f"\nTotal lines generated: {len(clean.split(chr(10)))}")

    print("\n" + "=" * 60)
    print("  COPILOT COMPLETE - Check tests/ folder!")
    print("=" * 60)

if __name__ == "__main__":
    run_copilot()