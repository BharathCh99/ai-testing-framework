import pytest
import ollama
import fitz
import time
import os

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF using pymupdf"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def ask_about_document(doc_text: str, question: str) -> dict:
    """Ask LLM questions about extracted document"""
    prompt = f"""You are a banking document analyzer.
Read the following document and answer the question.

DOCUMENT:
{doc_text}

QUESTION: {question}

Answer based ONLY on the document content."""

    start = time.time()
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    end = time.time()

    return {
        "answer": response['message']['content'],
        "response_time": round(end - start, 2)
    }

BASE_PATH = os.path.join(
    os.path.dirname(__file__),
    '..', 'test_documents')

# ── Bank Statement Tests ─────────────────────────────────

def test_extract_account_holder_name():
    """AI should extract account holder name from statement"""
    pdf_path = os.path.join(BASE_PATH, "bank_statement.pdf")
    text = extract_pdf_text(pdf_path)
    result = ask_about_document(
        text, "What is the account holder name?")
    answer = result['answer'].lower()
    assert "ahmad" in answer, \
        f"Name not found in: {answer[:200]}"
    print(f"\nAnswer: {result['answer'][:100]}")
    print(f"Response time: {result['response_time']}s")

def test_extract_account_balance():
    """AI should extract current balance"""
    pdf_path = os.path.join(BASE_PATH, "bank_statement.pdf")
    text = extract_pdf_text(pdf_path)
    result = ask_about_document(
        text, "What is the current account balance?")
    answer = result['answer']
    assert "15,432" in answer or "15432" in answer, \
        f"Balance not found in: {answer[:200]}"
    print(f"\nAnswer: {result['answer'][:100]}")

def test_extract_statement_period():
    """AI should identify statement period"""
    pdf_path = os.path.join(BASE_PATH, "bank_statement.pdf")
    text = extract_pdf_text(pdf_path)
    result = ask_about_document(
        text, "What is the statement period?")
    answer = result['answer'].lower()
    assert "2026" in answer or "may" in answer, \
        f"Period not found in: {answer[:200]}"
    print(f"\nAnswer: {result['answer'][:100]}")

# ── Loan Application Tests ───────────────────────────────

def test_extract_loan_amount():
    """AI should extract loan amount from application"""
    pdf_path = os.path.join(BASE_PATH, "loan_application.pdf")
    text = extract_pdf_text(pdf_path)
    result = ask_about_document(
        text, "What is the loan amount applied for?")
    answer = result['answer']
    assert "50,000" in answer or "50000" in answer, \
        f"Loan amount not found in: {answer[:200]}"
    print(f"\nAnswer: {result['answer'][:100]}")

def test_extract_loan_applicant():
    """AI should extract applicant name"""
    pdf_path = os.path.join(BASE_PATH, "loan_application.pdf")
    text = extract_pdf_text(pdf_path)
    result = ask_about_document(
        text, "What is the applicant name?")
    answer = result['answer'].lower()
    assert "siti" in answer, \
        f"Applicant not found in: {answer[:200]}"
    print(f"\nAnswer: {result['answer'][:100]}")

def test_extract_loan_status():
    """AI should identify loan approval status"""
    pdf_path = os.path.join(BASE_PATH, "loan_application.pdf")
    text = extract_pdf_text(pdf_path)
    result = ask_about_document(
        text, "What is the loan application status?")
    answer = result['answer'].lower()
    assert "approved" in answer, \
        f"Status not found in: {answer[:200]}"
    print(f"\nAnswer: {result['answer'][:100]}")

# ── KYC Form Tests ───────────────────────────────────────

def test_kyc_risk_rating():
    """AI should extract KYC risk rating"""
    pdf_path = os.path.join(BASE_PATH, "kyc_form.pdf")
    text = extract_pdf_text(pdf_path)
    result = ask_about_document(
        text, "What is the customer risk rating?")
    answer = result['answer'].lower()
    assert "low" in answer, \
        f"Risk rating not found in: {answer[:200]}"
    print(f"\nAnswer: {result['answer'][:100]}")

def test_kyc_pep_status():
    """AI should identify PEP status"""
    pdf_path = os.path.join(BASE_PATH, "kyc_form.pdf")
    text = extract_pdf_text(pdf_path)
    result = ask_about_document(
        text, "Is this customer a Politically Exposed Person?")
    answer = result['answer'].lower()
    assert "not" in answer or "no" in answer, \
        f"PEP status not found in: {answer[:200]}"
    print(f"\nAnswer: {result['answer'][:100]}")

# ── Performance Tests ────────────────────────────────────

def test_pdf_extraction_speed():
    """PDF extraction should be fast"""
    pdf_path = os.path.join(BASE_PATH, "bank_statement.pdf")
    start = time.time()
    text = extract_pdf_text(pdf_path)
    end = time.time()
    extraction_time = round(end - start, 3)
    assert extraction_time < 5, \
        f"PDF extraction too slow: {extraction_time}s"
    assert len(text) > 50, "PDF text too short"
    print(f"\nExtraction time: {extraction_time}s")
    print(f"Characters extracted: {len(text)}")

def test_multimodal_response_time():
    """Full pipeline should complete under 300s"""
    pdf_path = os.path.join(BASE_PATH, "kyc_form.pdf")
    text = extract_pdf_text(pdf_path)
    result = ask_about_document(
        text, "Summarize this KYC document in one sentence.")
    assert result['response_time'] < 300, \
        f"Too slow: {result['response_time']}s"
    print(f"\nFull pipeline time: {result['response_time']}s")