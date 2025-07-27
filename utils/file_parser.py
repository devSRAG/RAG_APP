import os
from pypdf import PdfReader
from docx import Document
from unstructured.partition.auto import partition

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return extract_pdf(file_path)
    elif ext == ".docx":
        return extract_docx(file_path)
    elif ext == ".txt":
        return open(file_path, "r", encoding="utf-8").read()
    else:
        return extract_with_unstructured(file_path)

def extract_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join([page.extract_text() for page in reader.pages])

def extract_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_with_unstructured(file_path):
    elements = partition(filename=file_path)
    return "\n".join([e.text for e in elements if e.text])
