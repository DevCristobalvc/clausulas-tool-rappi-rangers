from config import INPUT_DIR, OUTPUT_DIR, TXT_DIR
from extract_text import extract_text_from_pdf
from analyze_contract import analyze_contract
from utils import save_json

def generate_txts_from_pdfs():
    pdf_files = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() == ".pdf"])
    if not pdf_files:
        print("⚠️ No se encontraron PDFs en", INPUT_DIR)
        return

    for pdf_path in pdf_files:
        print(f"Extrayendo texto de {pdf_path.name} ...")
        extract_text_from_pdf(pdf_path)
        print(f"  ✅ Guardado en data/txts/{pdf_path.stem}.txt")

def analyze_txts():
    txt_files = sorted([p for p in TXT_DIR.iterdir() if p.suffix.lower() == ".txt"])
    if not txt_files:
        print("⚠️ No se encontraron TXT en", TXT_DIR)
        return

    for txt_path in txt_files:
        print(f"Analizando {txt_path.name} ...")
        text = txt_path.read_text(encoding="utf-8")
        result = analyze_contract(text)

        output_data = {
            "archivo": txt_path.stem,
            "analisis": result
        }

        out_file = OUTPUT_DIR / f"{txt_path.stem}.json"
        save_json(out_file, output_data)
        print(f"  ✅ Guardado: {out_file}")

if __name__ == "__main__":
    # Paso 1: generar TXT desde PDFs
    generate_txts_from_pdfs()

    # Paso 2: analizar los TXT generados
    analyze_txts()
