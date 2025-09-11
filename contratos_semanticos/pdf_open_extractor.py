import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

try:
    # OpenAI SDK v1
    from openai import OpenAI
except Exception as import_error:  # pragma: no cover
    raise RuntimeError(
        "The 'openai' package is required. Install it with: pip install openai>=1.0.0"
    ) from import_error


def build_client() -> OpenAI:
    """Build an OpenAI-compatible client using environment variables.

    Expects the following env vars:
    - SUMMARY_BASE_URL: Base URL for the OpenAI-compatible API
    - SUMMARY_API_KEY: API key/token
    - SUMMARY_MODEL: Model name used when creating responses
    """
    base_url = os.getenv("SUMMARY_BASE_URL")
    api_key = os.getenv("SUMMARY_API_KEY")

    missing = []
    if not base_url:
        missing.append("SUMMARY_BASE_URL")
    if not api_key:
        missing.append("SUMMARY_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    return OpenAI(base_url=base_url, api_key=api_key)


def ask_pdf(
    pdf_path: str,
    question: str,
    model: Optional[str] = None,
) -> str:
    """Upload a PDF and ask a question using the Responses API.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        question: The user question prompt.
        model: Optional override for the model. Defaults to SUMMARY_MODEL.

    Returns:
        The text output from the response.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    summary_model = model or os.getenv("SUMMARY_MODEL")
    if not summary_model:
        raise EnvironmentError("Missing required environment variable: SUMMARY_MODEL")

    client = build_client()

    with open(pdf_path, "rb") as fp:
        uploaded = client.files.create(file=fp, purpose="user_data")

    response = client.responses.create(
        model=summary_model,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Eres un convertidor de PDF a TEXTO PLANO. REGLAS ESTRICTAS:\n"
                            "1) NO AGREGUES NINGUNA PALABRA, DATO, RESUMEN, TRADUCCIÓN NI COMENTARIO.\n"
                            "2) NO ESCRIBAS NADA DISTINTO AL TEXTO DEL PDF. CERO 'observaciones', 'resumen', 'nota', títulos añadidos o metacomentarios.\n"
                            "3) Devuélvelo en el MISMO ORDEN DE LECTURA, respetando saltos de línea y separación por secciones/páginas, SIN insertar rótulos nuevos.\n"
                            "4) NO completes ni infieras texto faltante. Si una parte está ilegible, déjala tal cual.\n"
                            "5) La salida debe ser únicamente el TEXTO COMPLETO del documento, nada más."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": uploaded.id},
                    {"type": "input_text", "text": question},
                ],
            },
        ],
    )

    # The v1 SDK exposes convenience property output_text
    return getattr(response, "output_text", "")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a PDF and query it using an OpenAI-compatible Responses API.\n"
            "Uses SUMMARY_BASE_URL, SUMMARY_API_KEY, SUMMARY_MODEL from the environment.\n"
            "Defaults: --pdfs-dir to a 'pdfs' folder next to this script, and --output-dir to an 'outputs' folder next to this script if not provided."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument(
        "--pdf",
        dest="pdf",
        help="Path to a single PDF file (e.g., contratos_semanticos/pdfs/Contrato...pdf)",
    )
    source_group.add_argument(
        "--pdfs-dir",
        dest="pdfs_dir",
        help="Path to a directory containing PDF files to process in batch",
    )
    parser.add_argument(
        "--question",
        dest="question",
        required=False,
        default=None,
        help="Question to ask about the PDF(s). If omitted, uses a generic resumen.",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=None,
        help="Optional model override (otherwise uses SUMMARY_MODEL)",
    )
    # Defaults similar to extractor_ollama_resume.py style (relative to this file)
    script_dir = Path(__file__).resolve().parent
    default_pdfs_dir = str(script_dir / "pdfs")
    default_output_dir = str(script_dir / "outputs")
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=default_output_dir,
        help=f"Directory to write outputs (default: {default_output_dir})",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    # Carga variables de entorno desde .env (si existe), para comportamiento similar a extractor_ollama_resume.py
    load_dotenv()
    args = parse_args(argv)
    question = args.question or (
        "DEVUELVE SOLO EL TEXTO LITERAL COMPLETO DEL PDF, por cada sección/página en orden. "
        "NO agregues nada, NO resumas, NO comentes, NO traduzcas, NO inventes."
    )

    try:
        if args.pdf:
            answer = ask_pdf(pdf_path=args.pdf, question=question, model=args.model)
            print(answer)
            return 0

        # Batch mode - use default folder next to this script if not provided
        script_dir = Path(__file__).resolve().parent
        pdfs_dir = args.pdfs_dir or str(script_dir / "pdfs")
        if not os.path.isdir(pdfs_dir):
            raise NotADirectoryError(f"Directory not found: {pdfs_dir}")

        output_dir = args.output_dir or str(script_dir / "outputs")
        os.makedirs(output_dir, exist_ok=True)

        processed_files: List[str] = []
        for entry in sorted(os.listdir(pdfs_dir)):
            if not entry.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(pdfs_dir, entry)
            try:
                answer = ask_pdf(pdf_path=pdf_path, question=question, model=args.model)
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
    raise SystemExit(main(sys.argv[1:]))


