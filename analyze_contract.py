import json
import logging
from openai import OpenAI
from pydantic import ValidationError

from config import MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL
from models import ContratoAnalisis

# Config logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Cliente OpenAI
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

# Prompt del sistema (se puede mover a prompts.py)
SYSTEM_PROMPT = """
Eres un analista legal experto.
Analiza el texto de un contrato y devuelve SOLO un JSON con la siguiente estructura (campos exactos):
{ ... igual al que ya definiste ... }
No incluyas explicaciones ni texto extra fuera del JSON.
"""

def _extract_output_text(resp) -> str:
    """Extrae texto de la respuesta del modelo, manejando diferentes formatos"""
    output_text = ""
    try:
        for item in getattr(resp, "output", []):
            contents = getattr(item, "content", None) or item.get("content", [])
            for c in contents:
                if isinstance(c, dict) and "text" in c:
                    output_text += c["text"]
                elif hasattr(c, "text"):
                    output_text += c.text
    except Exception:
        # fallback: si existe atributo "output_text"
        output_text = getattr(resp, "output_text", "")
    return output_text.strip()

def analyze_contract(text: str) -> dict:
    """Envía el texto al modelo y devuelve un dict validado"""
    try:
        resp = client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            max_output_tokens=3000,
        )

        output_text = _extract_output_text(resp)

        # Intentar parsear a JSON
        parsed_json = json.loads(output_text)

        # Validar con Pydantic
        try:
            contrato = ContratoAnalisis(**parsed_json)
            return contrato.model_dump()
        except ValidationError as ve:
            logging.error(f"❌ Validación fallida: {ve}")
            return {}

    except json.JSONDecodeError as je:
        logging.error(f"❌ Error al parsear JSON: {je}")
        return {}
    except Exception as e:
        logging.error(f"❌ Análisis falló: {e}")
        return {}
