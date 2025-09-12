# -*- coding: utf-8 -*-
"""
Generador de resúmenes para campos específicos de contratos JSON.
- Resúmenes en claves *_Resumen (no sobreescribe el original).
- Chunking + fusión para textos largos.
- Salida de LLM: SOLO viñetas (ideas [General] y [Específico]).
"""

import os
import re
import json
import glob
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from dotenv import load_dotenv

load_dotenv()

# ===== Config requerido desde entorno (sin defaults) =====
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")
OLLAMA_LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL")

if not OLLAMA_BASE_URL or not OLLAMA_LLM_MODEL:
    raise RuntimeError("Faltan variables de entorno: OLLAMA_BASE_URL y/o OLLAMA_LLM_MODEL")

OLLAMA_BASE_URL = OLLAMA_BASE_URL.rstrip("/")

# Directorios
OUTPUT_EMBEDDINGS_DIR = Path("json_embeddings")
OUTPUT_RESUME_DIR = Path("json_with_resume")

# Campos a resumir -> (origen, destino)
CAMPOS_A_RESUMIR: List[Tuple[str, str]] = [
    ("DesembolsosPagos.DetalleDesembolsos", "DesembolsosPagos.DetalleDesembolsos_Resumen"),
    ("Exclusividad.CondicionesExclusividad", "Exclusividad.CondicionesExclusividad_Resumen"),
    ("Exclusividad.DetalleExclusividad", "Exclusividad.DetalleExclusividad_Resumen"),
    ("TerminoContrato.DetalleTermino", "TerminoContrato.DetalleTermino_Resumen"),
    ("VigenciaRenovacion.DetalleVigencia", "VigenciaRenovacion.DetalleVigencia_Resumen"),
]

# Chunking
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

# ---------- Cliente Ollama ----------
def ollama_generate(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": OLLAMA_LLM_MODEL, "prompt": prompt, "stream": False},
    )
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()

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
            r = ollama_generate(prompt_resumen(texto_limpio, campo_base, subcampo))
            return normalizar_vinetas(r)
        except Exception as e:
            return f"Error en la llamada a Ollama: {e}"

    # Map-Reduce para largo
    parciales: List[str] = []
    for i in range(0, len(texto_limpio), MAX_CHARS_PER_CHUNK):
        chunk = texto_limpio[i : i + MAX_CHARS_PER_CHUNK]
        try:
            r = ollama_generate(prompt_resumen(chunk, campo_base, subcampo))
        except Exception as e:
            r = f"- [General] Error chunk {len(parciales)+1}: {e}"
        parciales.append(normalizar_vinetas(r))

    try:
        fusion = ollama_generate(prompt_fusion(parciales, campo_base, subcampo))
        return normalizar_vinetas(fusion)
    except Exception as e:
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
    print(f"Ollama: {OLLAMA_BASE_URL} | Modelo: {OLLAMA_LLM_MODEL}")
    archivos = glob.glob(str(OUTPUT_EMBEDDINGS_DIR / "*.json"))
    print(f"JSON a procesar: {len(archivos)}")
    for i, archivo in enumerate(archivos, 1):
        print(f"[{i}/{len(archivos)}] {os.path.basename(archivo)}")
        procesar_archivo(archivo, str(OUTPUT_RESUME_DIR))
    print("OK")

if __name__ == "__main__":
    main()