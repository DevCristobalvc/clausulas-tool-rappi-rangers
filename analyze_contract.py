# analyze_contract.py
import json
import logging
import re
from pathlib import Path
from openai import OpenAI
from pydantic import ValidationError

from config import MODEL_NAME, OPENAI_BASE_URL, OPENAI_API_KEY
from models import ContratoAnalisis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """
Eres un analista legal experto especializado en contratos comerciales.
Analiza el documento proporcionado y extrae información específica sobre desembolsos, exclusividad, términos y vigencia.

Para cada campo booleano (TieneDesembolsos, TieneExclusividad, TerminacionUnilateral, PenalidadTerminacion, TieneRenovacionAutomatica), debes proporcionar:
- valor: true/false según si la cláusula existe
- evidencia: el texto exacto del contrato que respalda tu conclusión
- ubicacion: la sección, artículo o página donde se encuentra la información
- confianza: "alta", "media" o "baja" según qué tan clara es la evidencia

 IMPORTANTE:
 - Si el valor es false, en el campo "evidencia" escribe una explicación breve indicando que no se encontró nada relacionado con ese tema en el texto analizado.
 - Sigue esta guía para clasificar campos de acuerdo con data.md:
   1) Desembolsos y Pagos
      - PeriodicidadPagos: "Mensual", "Quincenal", "Contra Hitos" o "Otro".
      - CondicionesPago: texto libre con condiciones específicas.
      - FormaPago: "Transferencia", "Efectivo" u "Otro".
      - DetalleDesembolsos: texto completo explicativo extraído del contrato.
   2) Exclusividad
      - TieneExclusividad: true/false.
      - AlcanceExclusividad: "Territorial", "Producto/Servicio", "Clientes", "Otro" o "NA".
      - CondicionesExclusividad: texto libre con explicación.
      - RupturaExclusividad: incluir cómo puede romperse, si aplica.
      - DetalleExclusividad: texto completo explicativo extraído del contrato.
   3) Término del Contrato
      - DuracionContrato: número de meses/años o "Indefinido".
      - FechaInicio/FechaFin: formato YYYY-MM-DD si está explícita; si no, null.
      - TerminacionUnilateral y PenalidadTerminacion: true/false con evidencia.
      - Preaviso: número de días requeridos.
      - DetalleTermino: texto completo explicativo extraído del contrato.
   4) Vigencia y Renovación
      - TieneRenovacionAutomatica: true/false con evidencia.
      - PeriodicidadRenovacion: "Anual", "Semestral", "Otro" o "NA".
      - PreavisoNoRenovacion: número de días de preaviso o "NA".
      - DetalleVigencia: texto completo explicativo extraído del contrato.

Para campos de texto, proporciona la información específica encontrada en el contrato.

Devuelve ÚNICAMENTE un JSON válido, bien formado, que siga exactamente esta estructura:
{
  "TieneDesembolsos": {
    "valor": true/false,
    "evidencia": "texto del contrato",
    "ubicacion": "sección/artículo",
    "confianza": "alta/media/baja"
  },
  "TieneExclusividad": { ... },
  "TerminacionUnilateral": { ... },
  "PenalidadTerminacion": { ... },
  "TieneRenovacionAutomatica": { ... },
  "PeriodicidadPagos": "texto específico",
  "FormaPago": "texto específico",
  "CondicionesPago": "texto específico",
  "DetalleDesembolsos": "texto específico",
  "AlcanceExclusividad": "texto específico",
  "RupturaExclusividad": "texto específico",
  "CondicionesExclusividad": "texto específico",
  "DetalleExclusividad": "texto específico",
  "DuracionContrato": "texto específico",
  "FechaInicio": "fecha o null",
  "FechaFin": "fecha o null",
  "Preaviso": "texto específico",
  "DetalleTermino": "texto específico",
  "PeriodicidadRenovacion": "texto específico",
  "PreavisoNoRenovacion": "texto específico",
  "DetalleVigencia": "texto específico"
}

Si no encuentras información para un campo, usa:
- Para objetos Evidencia: {"valor": false, "evidencia": "", "ubicacion": "", "confianza": "baja"}
- Para strings: ""
- Para fechas: null

No incluyas comentarios, explicaciones, ni texto fuera del objeto JSON.
Devuelve un único bloque JSON cerrado con { }.
"""

DEBUG_DIR = Path("data/debugs")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

def _extract_output_text(resp) -> str:
    """
    Extrae de forma robusta el texto de la respuesta del modelo.
    Soporta varias formas: resp.output_text, resp.output (lista de dicts u objetos).
    """
    # 1) output_text si existe (forma simple)
    if hasattr(resp, "output_text") and getattr(resp, "output_text"):
        return str(getattr(resp, "output_text")).strip()

    # 2) intentar usar resp.output (puede ser lista de dicts o lista de objetos)
    output = ""
    out = None
    try:
        out = getattr(resp, "output", None)
    except Exception:
        out = None
    # si todavía no hay, intentar acceso como dict
    if out is None:
        try:
            out = resp["output"]  # some SDKs allow dict-like access
        except Exception:
            out = None

    if not out:
        # último recurso, serializar repr
        try:
            return repr(resp)
        except Exception:
            return ""

    for item in out:
        # item puede ser dict o un objeto con atributos
        contents = None
        if isinstance(item, dict):
            contents = item.get("content") or item.get("message") or item.get("output")
        else:
            contents = getattr(item, "content", None) or getattr(item, "message", None) or getattr(item, "output", None)

        # si contents es string/None/iterable
        if isinstance(contents, str):
            output += contents
            continue
        if not contents:
            # revisar si item tiene .text o "text"
            if isinstance(item, dict) and "text" in item:
                output += item["text"]
            else:
                txt = getattr(item, "text", None)
                if isinstance(txt, str):
                    output += txt
            continue

        # contents es iterable (lista)
        for c in contents:
            if isinstance(c, dict):
                # forma típica: {"text": "..."}
                if "text" in c and isinstance(c["text"], str):
                    output += c["text"]
                else:
                    # concatenar cualquier value string que aparezca
                    for v in c.values():
                        if isinstance(v, str):
                            output += v
            else:
                # c puede ser objeto con .text
                txt = getattr(c, "text", None) or getattr(c, "content", None)
                if isinstance(txt, str):
                    output += txt

    return output.strip()

def _sanitize_json(raw_text: str) -> str:
    """
    Limpia el raw_text intentando dejar un bloque JSON válido:
    - reemplaza comillas “curly” por comillas normales
    - extrae el primer bloque { ... }
    - quita comas colgantes antes de } o ]
    """
    if not raw_text:
        return "{}"
    s = raw_text.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("—", "-")

    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        s = m.group(0)

    # quitar comas colgantes
    s = re.sub(r",\s*([\}\]])", r"\1", s)

    # balancear llaves
    open_braces = s.count("{")
    close_braces = s.count("}")
    if close_braces < open_braces:
        s += "}" * (open_braces - close_braces)

    return s

def _fill_missing(parsed_json: dict) -> dict:
    """Rellena claves faltantes con defaults desde ContratoAnalisis"""
    base = ContratoAnalisis().model_dump()
    if not isinstance(parsed_json, dict):
        return base
    
    # Para campos de evidencia, asegurar que tengan la estructura correcta
    evidencia_fields = ['TieneDesembolsos', 'TieneExclusividad', 'TerminacionUnilateral', 
                       'PenalidadTerminacion', 'TieneRenovacionAutomatica']
    
    for field in evidencia_fields:
        if field in parsed_json:
            if isinstance(parsed_json[field], dict):
                # Si ya es un dict, asegurar que tenga todos los campos necesarios
                evidencia_default = {"valor": False, "evidencia": "", "ubicacion": "", "confianza": "baja"}
                evidencia_default.update(parsed_json[field])
                parsed_json[field] = evidencia_default
            else:
                # Si es un booleano simple, convertir a estructura de evidencia
                parsed_json[field] = {
                    "valor": bool(parsed_json[field]),
                    "evidencia": "",
                    "ubicacion": "",
                    "confianza": "baja"
                }
    
    base.update(parsed_json)
    # Asegurar explicación por defecto cuando valor es false y evidencia está vacía
    default_no_evidence_msg = "No se encontró información relacionada con este tema en el contrato analizado."
    for field in evidencia_fields:
        ev = base.get(field)
        if isinstance(ev, dict):
            if not ev.get("valor") and not ev.get("evidencia"):
                ev["evidencia"] = default_no_evidence_msg
                # Mantener ubicacion vacía y confianza baja si no hay evidencia
                if not ev.get("ubicacion"):
                    ev["ubicacion"] = ""
                if not ev.get("confianza"):
                    ev["confianza"] = "baja"
                base[field] = ev

    return base

def analyze_contract(text: str, file_name: str = "debug") -> dict:
    """
    Envía el texto al LLM, guarda raw para debug, intenta sanear y parsear el JSON,
    valida con Pydantic y devuelve siempre un dict con todos los campos.
    """
    try:
        resp = client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            #max_output_tokens=3000,
        )

        raw_output = _extract_output_text(resp)
        # Guardar raw para inspección
        (DEBUG_DIR / f"{file_name}.raw.txt").write_text(raw_output, encoding="utf-8")

        sanitized = _sanitize_json(raw_output)
        (DEBUG_DIR / f"{file_name}.sanitized.txt").write_text(sanitized, encoding="utf-8")

        # Intentar parsear
        try:
            parsed = json.loads(sanitized)
        except json.JSONDecodeError as e:
            logging.error(f"❌ JSON parse error ({file_name}): {e}")
            # Intento adicional: si había otro bloque JSON dentro, reintentar
            m = re.search(r"\{.*\}", sanitized, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception as e2:
                    logging.error(f"❌ Segundo intento de parseo falló ({file_name}): {e2}")
                    return ContratoAnalisis().model_dump()
            else:
                return ContratoAnalisis().model_dump()

        # Rellenar claves faltantes y validar
        filled = _fill_missing(parsed)
        try:
            contrato = ContratoAnalisis(**filled)
            return contrato.model_dump()
        except ValidationError as ve:
            logging.error(f"❌ Validación fallida ({file_name}): {ve}")
            # intentar con la base + parsed (ya hicimos fill_missing); si aun así falla, fallback
            try:
                fallback = ContratoAnalisis().model_dump()
                fallback.update(parsed if isinstance(parsed, dict) else {})
                contrato = ContratoAnalisis(**fallback)
                return contrato.model_dump()
            except Exception as e:
                logging.error(f"❌ Fallback tras validación fallida también falló ({file_name}): {e}")
                return ContratoAnalisis().model_dump()

    except Exception as e:
        logging.error(f"❌ Error en {file_name}: {e}")
        return ContratoAnalisis().model_dump()
