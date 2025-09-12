# -*- coding: utf-8 -*-
"""
Generador de resúmenes para campos específicos de contratos JSON.
- Resúmenes en claves *_Resumen (no sobreescribe el original).
- Chunking + fusión para textos largos.
- Salida de LLM: SOLO viñetas (ideas [General] y [Específico]).
- Backend: OpenAI Responses API (modelo tomado de entorno).
"""

import os
import re
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from dotenv import load_dotenv
from openai import OpenAI  # pip install openai python-dotenv

load_dotenv()

# ===== Config requerido desde entorno (SIN defaults) =====
OPENAI_API_KEY = os.environ.get("SUMMARY_API_KEY")
OPENAI_BASE_URL = os.environ.get("SUMMARY_BASE_URL")
OPENAI_MODEL = os.environ.get("SUMMARY_MODEL")  # p.ej. gpt-5-mini

if not OPENAI_API_KEY or not OPENAI_MODEL:
    raise RuntimeError(
        "Faltan variables de entorno: OPENAI_API_KEY y/o OPENAI_MODEL "
        "(ej: OPENAI_MODEL=gpt-5-mini)."
    )

client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

# Directorios
OUTPUT_EMBEDDINGS_DIR = Path("json_embeddings")
OUTPUT_RESUME_DIR = Path("json_with_resume_oai")

# Campos a resumir -> (origen, destino)
CAMPOS_A_RESUMIR: List[Tuple[str, str]] = [
    ("DesembolsosPagos.DetalleDesembolsos", "DesembolsosPagos.DetalleDesembolsos_Resumen"),
    ("Exclusividad.CondicionesExclusividad", "Exclusividad.CondicionesExclusividad_Resumen"),
    ("Exclusividad.DetalleExclusividad", "Exclusividad.DetalleExclusividad_Resumen"),
    ("TerminoContrato.DetalleTermino", "TerminoContrato.DetalleTermino_Resumen"),
    ("VigenciaRenovacion.DetalleVigencia", "VigenciaRenovacion.DetalleVigencia_Resumen"),
]

# Chunking (puedes ajustar por entorno si quieres)
MAX_CHARS_PER_CHUNK = int(os.getenv("MAX_CHARS_PER_CHUNK", "3000"))
MAX_TOTAL_CHARS = int(os.getenv("MAX_TOTAL_CHARS", "16000"))

# Limpieza de ruido típico de PDFs
RUIDO_REGEXES = [
    r"^\s*INFORMACIÓN CONFIDENCIAL\s*$",
    r"^\s*Página\s+\d+\s+de\s+\d+\s*$",
    r"^\s*\d+\s*-\s*\d+\s*$",
    r"^\s*\[\s*HOJA DE FIRMAS.*$",
]

# ---------- Utilidades JSON ----------
def get_nested_value(data: Dict[str, Any], path: str) -> Optional[Any]:
    cur = data
    try:
        for k in path.split("."):
            cur = cur[k]
        return cur
    except (KeyError, TypeError):
        return None

def ensure_path_and_set(data: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur = data
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

# ---------- Limpieza ----------
def limpiar_texto(texto: str) -> str:
    if len(texto) > MAX_TOTAL_CHARS:
        texto = texto[:MAX_TOTAL_CHARS] + "..."
    lineas = []
    for linea in texto.splitlines():
        l = linea.strip()
        if not l:
            continue
        if any(re.search(rgx, l, flags=re.IGNORECASE) for rgx in RUIDO_REGEXES):
            continue
        lineas.append(l)
    texto_limpio = " ".join(lineas)
    texto_limpio = re.sub(r"\s+", " ", texto_limpio).strip()
    return texto_limpio

def es_irrelevante(o: Any) -> bool:
    if o is None:
        return True
    if isinstance(o, str):
        s = o.strip().lower()
        return s in {"", "na", "n/a", "no aplica", "no hay nada significativo.", "no", "sin información"}
    return False

# ---------- Cliente OpenAI ----------
def openai_generate(prompt: str) -> str:
    # Responses API (salida en response.output_text)
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )
    return (getattr(resp, "output_text", "") or "").strip()

# ---------- Prompts ----------
def instruccion_para(campo_base: str) -> str:
    if campo_base == "DesembolsosPagos":
        return "condiciones de pago, periodicidad, descuentos/compensaciones, forma de pago"
    if campo_base == "Exclusividad":
        return "condiciones de exclusividad, alcance, sanciones/efectos por incumplimiento"
    if campo_base == "TerminoContrato":
        return "causales de terminación, preavisos, penalidades, derechos de las partes"
    if campo_base == "VigenciaRenovacion":
        return "vigencia, renovaciones automáticas, preavisos, penalidades asociadas"
    return "puntos más importantes"

def prompt_resumen(texto: str, campo_base: str, subcampo: str) -> str:
    # Solo viñetas. Orden: generales y específicos.
    return f"""
Eres un asistente legal. Lee el TEXTO FUENTE y devuelve ÚNICAMENTE viñetas (una por línea, empezando con "- ").

Campo: {campo_base}.{subcampo}
Enfoque: {instruccion_para(campo_base)}.

Formato de salida OBLIGATORIO:
- [General] ...
- [General] ...
- [General] ...
- [Específico] ... (incluye %/plazos/cifras si existen)
- [Específico] ...
- [Específico] ...

Reglas:
- Máximo 6 viñetas en total (3 generales + 3 específicas). Si no hay suficientes datos, da menos.
- Sin títulos, sin texto extra, sin explicaciones fuera de las viñetas.
- Cada viñeta: máx. 20 palabras, claras y concisas.

TEXTO FUENTE:
\"\"\"{texto}\"\"\"
""".strip()

def prompt_fusion(resumenes: List[str], campo_base: str, subcampo: str) -> str:
    lista = "\n".join(f"- {r}" for r in resumenes if r.strip())
    return f"""
Fusiona viñetas repetidas/solapadas y devuelve SOLO viñetas finales (máx. 6: 3 [General] + 3 [Específico]).
Mantén cifras, % y plazos relevantes. Sin texto fuera de las viñetas.

Viñetas parciales:
{lista}
""".strip()

# ---------- Generación ----------
def generar_resumen_ollama(texto: str, campo_path: str) -> str:
    """
    Conservamos el nombre para no tocar el resto del pipeline,
    pero ahora usa OpenAI en lugar de Ollama.
    """
    if es_irrelevante(texto) or (isinstance(texto, str) and texto.startswith("model=")):
        return "No hay información disponible."

    campo_base, subcampo = campo_path.split(".")
    texto_limpio = limpiar_texto(str(texto))
    if es_irrelevante(texto_limpio):
        return "No hay información disponible."

    def normalizar_vinetas(s: str) -> str:
        # Deja solo líneas con viñetas y corta a 6
        lineas = [l.strip() for l in s.splitlines()]
        v = [l for l in lineas if l.startswith("- ")]
        if not v:
            # fallback: intenta crear viñetas separando por "; "
            v = [f"- {p.strip()}" for p in re.split(r"[;\n]\s*", s) if p.strip()]
        # Limita a 6
        v = v[:6]
        # recorta cada viñeta a ~20 palabras
        def recortar(l: str, maxw: int = 20) -> str:
            palabras = l.split()
            return " ".join(palabras[:maxw])
        v = [recortar(l) for l in v]
        return "\n".join(v) if v else "No hay información disponible."

    # Un solo bloque
    if len(texto_limpio) <= MAX_CHARS_PER_CHUNK:
        try:
            r = openai_generate(prompt_resumen(texto_limpio, campo_base, subcampo))
            return normalizar_vinetas(r)
        except Exception as e:
            return f"Error al llamar OpenAI: {e}"

    # Map-Reduce para largo
    parciales: List[str] = []
    for i in range(0, len(texto_limpio), MAX_CHARS_PER_CHUNK):
        chunk = texto_limpio[i : i + MAX_CHARS_PER_CHUNK]
        try:
            r = openai_generate(prompt_resumen(chunk, campo_base, subcampo))
        except Exception as e:
            r = f"- [General] Error chunk {len(parciales)+1}: {e}"
        parciales.append(normalizar_vinetas(r))

    try:
        fusion = openai_generate(prompt_fusion(parciales, campo_base, subcampo))
        return normalizar_vinetas(fusion)
    except Exception:
        # Fallback: une y corta
        unido = "\n".join(parciales)
        return normalizar_vinetas(unido)

# ---------- Procesamiento ----------
def procesar_archivo(ruta_archivo: str, directorio_salida: str) -> None:
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        data = json.load(f)

    for src_path, dst_path in CAMPOS_A_RESUMIR:
        valor = get_nested_value(data, src_path)
        ya = get_nested_value(data, dst_path)
        if ya and not es_irrelevante(ya):
            continue
        if es_irrelevante(valor):
            ensure_path_and_set(data, dst_path, "No hay información disponible.")
            continue
        resumen = generar_resumen_ollama(str(valor), src_path)
        ensure_path_and_set(data, dst_path, resumen)

    nombre_archivo = os.path.basename(ruta_archivo)
    nombre_sin_extension = os.path.splitext(nombre_archivo)[0]
    ruta_salida = os.path.join(directorio_salida, f"{nombre_sin_extension}_with_resume.json")

    os.makedirs(directorio_salida, exist_ok=True)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    print(f"OpenAI modelo: {OPENAI_MODEL}")
    archivos = glob.glob(str(OUTPUT_EMBEDDINGS_DIR / "*.json"))
    print(f"JSON a procesar: {len(archivos)}")
    for i, archivo in enumerate(archivos, 1):
        print(f"[{i}/{len(archivos)}] {os.path.basename(archivo)}")
        procesar_archivo(archivo, str(OUTPUT_RESUME_DIR))
    print("OK")

if __name__ == "__main__":
    main()