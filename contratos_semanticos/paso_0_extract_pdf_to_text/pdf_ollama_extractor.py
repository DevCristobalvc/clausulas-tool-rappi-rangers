import os
import sys
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv


"""
Extractor de PDFs usando modelos de visión de Ollama.
Ejecuta sin argumentos y procesa todos los PDFs en 'pdfs' junto a este script,
generando archivos .txt en 'outputs'.
"""


def _require_module(module_name: str, install_instr: str):
    try:
        return __import__(module_name)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Missing dependency '{module_name}'. Install it with: {install_instr}"
        ) from exc


def _render_pdf_pages_to_images(pdf_path: str, zoom: float = 2.0) -> List[bytes]:
    fitz = _require_module("fitz", "pip install PyMuPDF>=1.23.0")
    doc = fitz.open(pdf_path)
    images: List[bytes] = []
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        # Render at higher resolution and without alpha to help OCR
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        images.append(pix.tobytes("png"))
    return images


def _extract_pdf_pages_text(pdf_path: str) -> List[str]:
    fitz = _require_module("fitz", "pip install PyMuPDF>=1.23.0")
    doc = fitz.open(pdf_path)
    texts: List[str] = []
    for page in doc:
        text = page.get_text("text") or ""
        texts.append(text.strip())
    return texts


def ask_pdf_ollama(
    pdf_path: str,
    question: str,
    model: Optional[str] = None,
) -> str:
    """Process a PDF with an Ollama vision-capable model by rendering pages to images.

    Requires:
    - PyMuPDF (fitz) for page rendering
    - ollama Python client
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Prefer explicit model argument; otherwise try OLLAMA_MODEL; fallback to gemma3:latest
    ollama_model = model or os.getenv("OLLAMA_MODEL") or "gemma3:latest"

    ollama = _require_module("ollama", "pip install ollama>=0.3.0")

    page_texts = _extract_pdf_pages_text(pdf_path)
    outputs: List[str] = []
    system_prompt = (
        "Eres un convertidor de PDF a TEXTO PLANO. REGLAS ESTRICTAS:\n"
        "1) NO AGREGUES NINGUNA PALABRA, DATO, RESUMEN, TRADUCCIÓN NI COMENTARIO.\n"
        "2) NO ESCRIBAS NADA DISTINTO AL TEXTO DEL PDF. CERO 'observaciones', 'resumen', 'nota', títulos añadidos o metacomentarios.\n"
        "3) Devuélvelo en el MISMO ORDEN DE LECTURA de la página renderizada, respetando saltos de línea.\n"
        "4) NO completes ni infieras texto faltante. Si una parte está ilegible, déjala tal cual.\n"
        "5) La salida debe ser únicamente el TEXTO COMPLETO visible, nada más."
    )
    # Prepare images only if needed (lazy)
    images_cache: Optional[List[bytes]] = None

    for idx, page_text in enumerate(page_texts):
        if page_text:
            outputs.append(page_text)
            continue

        if images_cache is None:
            images_cache = _render_pdf_pages_to_images(pdf_path, zoom=2.0)
        if idx >= len(images_cache):
            outputs.append("")
            continue

        image_bytes = images_cache[idx]
        try:
            response = ollama.chat(
                model=ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            (question or "Extrae el TEXTO PLANO visible de esta página.")
                            + "\nSolo devuelve el texto reconocido, sin títulos adicionales."
                        ),
                        "images": [image_bytes],
                    },
                ],
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Ollama chat failed: {exc}") from exc

        content = ""
        if isinstance(response, dict):
            message = response.get("message") if hasattr(response, "get") else None
            if isinstance(message, dict):
                content = message.get("content", "")
            elif "content" in response:
                content = response.get("content", "")
        else:
            content = str(response)

        outputs.append(content or "")

    # Join page-wise outputs with a page separator to preserve order
    return "\n".join(outputs)
def main() -> int:
    # Carga variables de entorno desde .env (si existe)
    load_dotenv()

    # Pregunta por defecto alineada a extracción literal
    question = (
        "DEVUELVE SOLO EL TEXTO LITERAL COMPLETO DEL PDF, por cada sección/página en orden. "
        "NO agregues nada, NO resumas, NO comentes, NO traduzcas, NO inventes."
    )

    try:
        # Directorios por defecto junto a este script
        script_dir = Path(__file__).resolve().parent
        pdfs_dir = str(script_dir / "pdfs")
        if not os.path.isdir(pdfs_dir):
            raise NotADirectoryError(f"Directory not found: {pdfs_dir}")

        output_dir = str(script_dir / "outputs")
        os.makedirs(output_dir, exist_ok=True)

        processed_files: List[str] = []
        for entry in sorted(os.listdir(pdfs_dir)):
            if not entry.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(pdfs_dir, entry)
            try:
                answer = ask_pdf_ollama(pdf_path=pdf_path, question=question, model=os.getenv("OLLAMA_MODEL"))
            except Exception as file_exc:  # pragma: no cover
                print(f"Error processing {pdf_path}: {file_exc}", file=sys.stderr)
                continue

            base_name, _ = os.path.splitext(entry)
            out_path = os.path.join(output_dir, f"{base_name}.txt")
            try:
                with open(out_path, "w", encoding="utf-8") as out_fp:
                    out_fp.write(answer or "")
                processed_files.append(out_path)
            except Exception as write_exc:  # pragma: no cover
                print(f"Error writing {out_path}: {write_exc}", file=sys.stderr)

        print("\n".join(processed_files))
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


