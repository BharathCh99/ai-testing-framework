from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import os

os.makedirs("test_documents", exist_ok=True)

styles = getSampleStyleSheet()

# ── Document 1: Bank Statement ───────────────────────────
def create_bank_statement():
    doc = SimpleDocTemplate(
        "test_documents/bank_statement.pdf",
        pagesize=A4)
    elements = []

    elements.append(Paragraph(
        "RHB BANK BERHAD", styles['Title']))
    elements.append(Paragraph(
        "Account Statement", styles['Heading1']))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph(
        "Account Holder: Ahmad bin Abdullah", styles['Normal']))
    elements.append(Paragraph(
        "Account Number: 1234-5678-9012", styles['Normal']))
    elements.append(Paragraph(
        "Statement Period: 01/05/2026 - 31/05/2026",
        styles['Normal']))
    elements.append(Paragraph(
        "Current Balance: RM 15,432.50", styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))

    data = [
        ['Date', 'Description', 'Debit(RM)', 'Credit(RM)', 'Balance(RM)'],
        ['01/05', 'Opening Balance', '', '', '12,000.00'],
        ['03/05', 'Salary Credit', '', '5,000.00', '17,000.00'],
        ['05/05', 'ATM Withdrawal', '500.00', '', '16,500.00'],
        ['10/05', 'Online Transfer', '1,200.00', '', '15,300.00'],
        ['15/05', 'EPF Contribution', '550.00', '', '14,750.00'],
        ['20/05', 'Interest Credit', '', '682.50', '15,432.50'],
    ]

    table = Table(data, colWidths=[1*inch, 2.5*inch,
                                    1*inch, 1*inch, 1*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.white, colors.lightgrey]),
        ('ALIGN', (2,0), (-1,-1), 'RIGHT'),
    ]))
    elements.append(table)
    doc.build(elements)
    print("✅ Created: bank_statement.pdf")

# ── Document 2: Loan Application ────────────────────────
def create_loan_application():
    doc = SimpleDocTemplate(
        "test_documents/loan_application.pdf",
        pagesize=A4)
    elements = []

    elements.append(Paragraph(
        "RHB BANK - PERSONAL LOAN APPLICATION",
        styles['Title']))
    elements.append(Spacer(1, 0.2*inch))

    data = [
        ['Field', 'Details'],
        ['Applicant Name', 'Siti binti Rahman'],
        ['IC Number', '850101-14-5678'],
        ['Date of Birth', '01 January 1985'],
        ['Employment', 'Senior Engineer, Petronas'],
        ['Monthly Salary', 'RM 8,500'],
        ['Loan Amount', 'RM 50,000'],
        ['Loan Tenure', '5 Years'],
        ['Purpose', 'Home Renovation'],
        ['Interest Rate', '5.99% per annum'],
        ['Monthly Payment', 'RM 966.64'],
    ]

    table = Table(data, colWidths=[2.5*inch, 3.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.white, colors.lightgrey]),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph(
        "Status: APPROVED - Pending Document Verification",
        styles['Heading2']))
    doc.build(elements)
    print("✅ Created: loan_application.pdf")

# ── Document 3: KYC Form ─────────────────────────────────
def create_kyc_form():
    doc = SimpleDocTemplate(
        "test_documents/kyc_form.pdf",
        pagesize=A4)
    elements = []

    elements.append(Paragraph(
        "KYC VERIFICATION FORM", styles['Title']))
    elements.append(Paragraph(
        "Know Your Customer - Bank Negara Malaysia Compliance",
        styles['Heading2']))
    elements.append(Spacer(1, 0.2*inch))

    data = [
        ['KYC Field', 'Submitted Value', 'Status'],
        ['Full Name', 'Raj Kumar Krishnan', '✓ Verified'],
        ['Passport No', 'A12345678', '✓ Verified'],
        ['Nationality', 'Malaysian PR', '✓ Verified'],
        ['Address', 'No 12 Jalan Bukit Bintang KL', '✓ Verified'],
        ['Phone', '+60 12-345 6789', '✓ Verified'],
        ['Email', 'raj.kumar@email.com', '✓ Verified'],
        ['Source of Funds', 'Employment Income', '✓ Verified'],
        ['PEP Status', 'Not a PEP', '✓ Verified'],
        ['Risk Rating', 'LOW RISK', '✓ Approved'],
    ]

    table = Table(data, colWidths=[2*inch, 2.5*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('TEXTCOLOR', (2,1), (2,-1), colors.green),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.white, colors.lightgrey]),
    ]))
    elements.append(table)
    doc.build(elements)
    print("✅ Created: kyc_form.pdf")

# ── Run all ──────────────────────────────────────────────
print("=" * 50)
print("  Creating Banking Test Documents")
print("=" * 50)
create_bank_statement()
create_loan_application()
create_kyc_form()
print("\n✅ All documents created in test_documents/")