from pathlib import Path
import pdfplumber
from config import TXT_DIR

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extrae texto completo de un PDF y lo guarda en data/txts"""
    text_all = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_all.append(text)

    full_text = "\n".join(text_all)

    # Guardar como .txt
    txt_file = TXT_DIR / f"{pdf_path.stem}.txt"
    with txt_file.open("w", encoding="utf-8") as f:
        f.write(full_text)

    return full_text
