import pdfplumber

with pdfplumber.open("data/fabricinspectionsamplereport.pdf") as pdf:
    text = pdf.pages[1].extract_text()
    print(repr(text))