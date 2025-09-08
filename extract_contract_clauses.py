import os
import json
from pathlib import Path
import pdfplumber
from openai import OpenAI  # asumes el mismo cliente que usaste antes
from pydantic import BaseModel
from typing import List, Optional

from dotenv import load_dotenv


# Cargar .env
load_dotenv()

# -------------------------
# CONFIG (ajusta si hace falta)
# -------------------------
INPUT_DIR = Path("contratos")
OUTPUT_DIR = Path("salida")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Lee credenciales desde variables de entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
# Si prefieres, puedes construir client con parámetros directos (no recomendado para prod)
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

# -------------------------
# Pydantic models (para referencia)
# -------------------------
class ClauseOut(BaseModel):
    id: int
    text: str
    is_clause: bool
    title: Optional[str] = None
    summary: Optional[str] = None

class ClausesList(BaseModel):
    clauses: List[ClauseOut]

# -------------------------
# Utilidades: extracción y batching
# -------------------------
def extract_paragraphs_from_pdf(pdf_path: Path) -> List[str]:
    """Extrae texto y agrupa líneas en párrafos razonables."""
    paragraphs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            lines = [ln.rstrip() for ln in text.split("\n")]
            cur = ""
            for ln in lines:
                ln = ln.strip()
                if ln == "":
                    if cur:
                        paragraphs.append(cur.strip())
                        cur = ""
                else:
                    # manejar guiones al final de linea (hyphenation)
                    if cur.endswith("-"):
                        cur = cur[:-1] + ln
                    else:
                        if cur:
                            cur = cur + " " + ln
                        else:
                            cur = ln
            if cur:
                paragraphs.append(cur.strip())
    return paragraphs

def chunkify(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n], i

# -------------------------
# Conversación con el LLM
# -------------------------

#REVISAR: los tipos de pronts que se puede usar parar que el modelo entienda bien

SYSTEM_PROMPT = """
Eres un extractor experto en contratos. Te daré una lista de párrafos numerados.
Para cada párrafo debes decidir si se trata de una cláusula contractual (por ejemplo: obligaciones, pagos, plazos, indemnizaciones, vigencia, terminación, confidencialidad, etc.)
Devuelve **solo** UN JSON: una lista (array) de objetos con las siguientes claves:
- id (int): el número del párrafo según la entrada
- text (string): el texto original del párrafo
- is_clause (boolean): true si es una cláusula contractual relevante, false si es encabezado, índice, pie de página, tabla, nota o texto no contractual
- title (string|null): si puedes inferir un título corto de la cláusula (máx 6 palabras) o null
- summary (string|null): resumen en una línea (máx 20 palabras) o null

No agregues explicaciones, encabezados ni nada más: SOLO el JSON válido.
"""

def call_model_for_paragraphs(paragraphs: List[str], start_idx: int):
    """
    Llama al modelo para un batch de párrafos.
    Devuelve lista de diccionarios (según spec).
    """
    # Construir prompt del usuario: numerar cada párrafo con su id absoluto
    lines = []
    for i, p in enumerate(paragraphs):
        pid = start_idx + i + 1  # ids empiezan en 1
        # limitamos tamaño por seguridad (el modelo puede trabajar con textos largos,
        # pero si hay párrafos gigantes quizá convenga truncar o manejarlo).
        lines.append(f"{pid}) {p}")
    user_content = "\n\n".join(lines)

    # Intentamos usar responses.parse (si tu SDK/endpoint lo soporta)
    try:
        resp = client.responses.parse(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            text_format=ClausesList  # le pedimos mapear a nuestro esquema
        )
        # intentamos obtener el dict de salida
        if hasattr(resp, "output_parsed") and resp.output_parsed:
            # pydantic v2 usa model_dump
            try:
                parsed = resp.output_parsed.model_dump()
            except Exception:
                try:
                    parsed = resp.output_parsed.dict()
                except Exception:
                    parsed = None
            if parsed and isinstance(parsed, dict) and "clauses" in parsed:
                return parsed["clauses"]
    except Exception as e:
        # fallback a un path más simple
        print(f"[WARN] responses.parse falló: {e}. Intentando fallback JSON raw...")

    # Fallback: pedir al modelo que devuelva JSON en texto plano y parsearlo con json.loads
    # REVISAR
    try:
        resp2 = client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            max_output_tokens=2000
        )
        # Extraer texto del output (varias formas segun versión SDK)
        output_text = ""
        # la estructura típica es resp2.output -> lista -> item["content"] -> lista de dicts con "text"
        for item in getattr(resp2, "output", []):
            contents = getattr(item, "content", None) or item.get("content", [])
            for c in contents:
                if isinstance(c, dict) and "text" in c:
                    output_text += c["text"]
                elif hasattr(c, "text"):
                    output_text += c.text
        if not output_text:
            # Otro fallback: resp2.output_text si existe
            output_text = getattr(resp2, "output_text", "") or ""

        # A veces el modelo imprime espacios o líneas antes/después; intentar localizar el JSON
        output_text = output_text.strip()
        # parsear
        parsed_json = json.loads(output_text)
        # si lo que devuelve es la lista directa, retornarla
        if isinstance(parsed_json, list):
            return parsed_json
        # si devuelve {"clauses": [...]}
        if isinstance(parsed_json, dict) and "clauses" in parsed_json:
            return parsed_json["clauses"]
    except Exception as e:
        print(f"[ERROR] fallback JSON parse falló: {e}")

    # Si todo falla, como último recurso devolvemos heurística: marcar todos como clauses simples
    fallback = []
    for i, p in enumerate(paragraphs):
        pid = start_idx + i + 1
        fallback.append({"id": pid, "text": p, "is_clause": True, "title": None, "summary": None})
    return fallback

# -------------------------
# Flujo principal
# -------------------------
def process_all_contracts():
    pdf_files = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() == ".pdf"])
    if not pdf_files:
        print("No se encontraron PDFs en", INPUT_DIR)
        return

    for pdf_path in pdf_files:
        print(f"Procesando {pdf_path.name} ...")
        paragraphs = extract_paragraphs_from_pdf(pdf_path)
        print(f"  párrafos detectados: {len(paragraphs)}")

        # Batching: enviar por lotes de N párrafos para evitar tokens excesivos
        batch_size = 30
        all_clauses = []
        for batch, start_idx in chunkify(paragraphs, batch_size):
            print(f"   llamando LLM para párrafos {start_idx+1} a {start_idx+len(batch)} ...")
            result_items = call_model_for_paragraphs(batch, start_idx)
            if result_items:
                # normalizar cada item esperado
                for it in result_items:
                    # asegurar estructura mínima
                    try:
                        cid = int(it.get("id"))
                    except Exception:
                        cid = None
                    text = it.get("text") or ""
                    is_clause = bool(it.get("is_clause"))
                    title = it.get("title") if "title" in it else None
                    summary = it.get("summary") if "summary" in it else None
                    all_clauses.append({
                        "id": cid,
                        "text": text,
                        "is_clause": is_clause,
                        "title": title,
                        "summary": summary
                    })
            else:
                print("   [WARN] LLM devolvió vacío para este batch; usando fallback heurístico.")
                for i, p in enumerate(batch):
                    all_clauses.append({
                        "id": start_idx + i + 1,
                        "text": p,
                        "is_clause": True,
                        "title": None,
                        "summary": None
                    })

        # Filtrar solo cláusulas detectadas si quieres sólo las cláusulas
        detected_clauses = [c for c in all_clauses if c.get("is_clause")]

        output_data = {
            "archivo": pdf_path.name,
            "total_parrafos_detectados": len(paragraphs),
            "total_clausulas_extraidas": len(detected_clauses),
            "clausulas": detected_clauses,
            # opcional: puedes guardar también todos los items para revisión
            "raw_items": all_clauses
        }

        out_file = OUTPUT_DIR / f"{pdf_path.stem}.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"  ✅ Guardado: {out_file} (cláusulas: {len(detected_clauses)})\n")


if __name__ == "__main__":
    process_all_contracts()
