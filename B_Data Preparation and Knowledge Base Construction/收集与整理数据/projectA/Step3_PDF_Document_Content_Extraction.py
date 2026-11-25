from PyPDF2 import PdfReader
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
manual_text = extract_pdf_text("机床用户手册.pdf")