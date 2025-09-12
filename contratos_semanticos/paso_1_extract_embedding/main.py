# -*- coding: utf-8 -*-
"""
Script mejorado para extracción semántica de cláusulas de contratos con dos modos:
1. Modo embeddings: Extrae información usando embeddings y guarda en json_embeddings/
2. Modo resumen: Procesa archivos de json_embeddings/ y genera resúmenes en json_with_resume/

Mejoras basadas en feedback:
- DesembolsosPagos: Enfocado en pagos extraordinarios (bonos, upfronts, etc.) excluyendo órdenes
- Exclusividad: Mejor detección de exclusividad de plataforma vs otras plataformas
- TerminoContrato: Mejor extracción de condiciones de terminación unilateral
- Campos de detalle: Mejor preservación del contenido completo sin cortes
"""

import os
import re
import json
import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from dotenv import load_dotenv
from rapidfuzz import fuzz
import dateparser

# ==========================
# CONFIG
# ==========================

#INPUT_DIR = Path("input_txt")
INPUT_DIR = Path("outputs")
OUTPUT_EMBEDDINGS_DIR = Path("json_embeddings")
OUTPUT_RESUME_DIR = Path("json_with_resume")

load_dotenv()
# Configuración de Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1")

# Modelo embeddings (usando Ollama)
EMBEDDING_MODEL = OLLAMA_EMBED_MODEL

# Umbral de similitud para considerar un párrafo "relevante"
SIM_THRESHOLD = 0.45

# Límite conservador de caracteres por chunk
MAX_CHARS_PER_CHUNK = 8000

# Prompts/keywords mejorados por tema basados en feedback
THEME_QUERIES = {
    "DesembolsosPagos": [
        # Enfocado en pagos extraordinarios, no en órdenes regulares
        "bono de firma", "upfront", "bono de apertura", "pago inicial", "desembolso extraordinario",
        "compensación adicional", "pago único", "incentivo de firma", "pago por suscripción",
        "desembolso de suma", "pago de bienvenida", "bono por apertura", "pago inicial del contrato"
    ],
    "Exclusividad": [
        # Enfocado en exclusividad de plataforma
        "exclusividad", "acuerdo exclusivo", "no puede vender en otras plataformas",
        "prohibido usar otras plataformas", "exclusivo con rappi", "no competencia",
        "restricción de plataformas", "exclusividad de venta", "no puede usar uber eats",
        "no puede usar didi food", "exclusividad territorial", "exclusividad de producto"
    ],
    "TerminoContrato": [
        # Enfocado en terminación unilateral del aliado
        "terminación unilateral", "terminar el contrato", "dar por terminado",
        "rescindir el contrato", "terminación anticipada", "condiciones de terminación",
        "cómo terminar el contrato", "procedimiento de terminación", "causales de terminación",
        "terminación sin causa", "preaviso de terminación", "requisitos para terminar"
    ],
    "VigenciaRenovacion": [
        "vigencia del contrato", "duración del contrato", "plazo del contrato",
        "renovación automática", "renovación del contrato", "término y renovación",
        "fecha de vencimiento", "prórroga del contrato", "extensión del contrato"
    ]
}

# ==========================
# UTILIDADES (reutilizadas del extractor_ollama.py)
# ==========================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 1:
        a = a[np.newaxis, :]
    if b.ndim == 1:
        b = b[np.newaxis, :]
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
    return float((a_norm @ b_norm.T).max())

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _split_by_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[\.!?])\s+", text)
    return [re.sub(r"[ \t]+", " ", s).strip() for s in sentences if s and s.strip()]

def _safe_chunk_long_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = _split_by_sentences(text)
    parts: List[str] = []
    current = []
    current_len = 0
    for s in sentences:
        s_len = len(s) + 1
        if current_len + s_len <= max_chars or not current:
            current.append(s)
            current_len += s_len
        else:
            parts.append(" ".join(current).strip())
            current = [s]
            current_len = s_len
    if current:
        parts.append(" ".join(current).strip())
    final_parts: List[str] = []
    for part in parts:
        if len(part) <= max_chars:
            final_parts.append(part)
        else:
            for i in range(0, len(part), max_chars):
                final_parts.append(part[i:i+max_chars])
    return final_parts

def paragraph_chunk(text: str) -> List[str]:
    raw = re.split(r"\n{2,}", text)
    chunks: List[str] = []
    for p in raw:
        pp = re.sub(r"[ \t]+", " ", p).strip()
        pp = re.sub(r"^[•\-\u2022]+\s*", "", pp)
        if len(pp) >= 20:
            safe_parts = _safe_chunk_long_text(pp, MAX_CHARS_PER_CHUNK)
            chunks.extend(safe_parts)
    return chunks

def has_negation_near(term: str, text: str, window: int = 5) -> bool:
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    term_tokens = re.findall(r"\w+", term.lower(), flags=re.UNICODE)
    negs = {"no", "sin", "nunca"}
    for i in range(len(tokens)):
        if tokens[i:i+len(term_tokens)] == term_tokens:
            start = max(0, i - window)
            if any(t in negs for t in tokens[start:i+1]):
                return True
    return False

def text_or_na(s: Optional[str]) -> str:
    return s.strip() if s and s.strip() else "No hay nada significativo."

def pick_sentence_excerpt(text: str, keys: List[str], fallback_chars: int = 500) -> str:
    """Mejorado para preservar más contenido y evitar cortes"""
    sents = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    best = ""
    best_score = 0
    target = " ".join(keys).lower()
    for s in sents:
        score = fuzz.partial_ratio(target, s.lower())
        if score > best_score:
            best, best_score = s, score
    if best and best_score >= 50:
        return best.strip()
    # Aumentar el fallback para evitar cortes
    return text.strip()[:fallback_chars].strip()

# ------------------ Ollama Embeddings helpers ------------------

def _ollama_embed_batch(texts: List[str], model: str) -> np.ndarray:
    embs = []
    for t in texts:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                r = requests.post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    json={"model": model, "prompt": t},
                    timeout=120
                )
                r.raise_for_status()
                embs.append(np.array(r.json()["embedding"], dtype=np.float32))
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"Error al obtener embedding para texto (intento {attempt + 1}): {e}")
                    embs.append(np.zeros(768, dtype=np.float32))
                else:
                    print(f"Reintentando embedding (intento {attempt + 1}/{max_retries}): {e}")
                    continue
    return np.vstack(embs) if embs else np.zeros((0, 0), dtype=np.float32)

def embed_texts(client_unused, texts: List[str], model: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    return _ollama_embed_batch(texts, OLLAMA_EMBED_MODEL)

def embed_centroid(client_unused, queries: List[str], model: str) -> np.ndarray:
    if not queries:
        return np.zeros((1, 1), dtype=np.float32)
    vecs = _ollama_embed_batch(queries, OLLAMA_EMBED_MODEL)
    centroid = vecs.mean(axis=0)
    centroid = centroid / max(np.linalg.norm(centroid), 1e-8)
    return centroid

def best_paragraph_for_theme(client_unused,
                             chunks: List[str],
                             theme_queries: List[str],
                             model: str) -> Tuple[Optional[str], float]:
    if not chunks:
        return None, 0.0
    theme_vec = embed_centroid(client_unused, theme_queries, model)
    para_embs = embed_texts(client_unused, chunks, model)
    if para_embs.size == 0:
        return None, 0.0
    norms = np.linalg.norm(para_embs, axis=1, keepdims=True)
    para_embs = para_embs / np.clip(norms, 1e-8, None)
    sims = para_embs @ theme_vec.reshape(-1, 1)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx, 0])
    return chunks[best_idx], best_sim

# ------------------ Extracciones mejoradas basadas en feedback ------------------

def find_bool_from_text(text: str, positive_cues: List[str], negative_cues: List[str]) -> str:
    t = text.lower()
    for n in negative_cues:
        if n in t or has_negation_near(n, text):
            return "NO"
    for p in positive_cues:
        if p in t and not has_negation_near(p, text):
            return "SI"
    return "NO"

def extract_periodicidad_pagos(text: str) -> str:
    t = text.lower()
    opciones = [
        "mensual", "quincenal", "semanal", "bimensual", "trimestral",
        "contra hitos", "por hitos", "a la entrega", "único", "una sola vez"
    ]
    for o in opciones:
        if o in t:
            return o.capitalize()
    return "NA"

def extract_forma_pago(text: str) -> str:
    t = text.lower()
    if "transferencia" in t:
        return "Transferencia"
    if "efectivo" in t:
        return "Efectivo"
    if "cheque" in t:
        return "Cheque"
    if "pse" in t:
        return "PSE"
    return "Otro"

def extract_duracion(text: str) -> str:
    t = text.lower()
    m = re.search(r"(\d+)\s*(mes|meses|año|años)", t)
    if m:
        num, unidad = m.group(1), m.group(2)
        return f"{num} {unidad}"
    if "indefinid" in t:
        return "Indefinido"
    return "NA"

def normalize_date(d: str) -> Optional[str]:
    dt = dateparser.parse(d, languages=["es", "en"])
    return dt.strftime("%Y-%m-%d") if dt else None

def extract_dates(text: str) -> Tuple[Optional[str], Optional[str]]:
    iso_pat = r"\b(\d{4}-\d{2}-\d{2})\b"
    found = re.findall(iso_pat, text)
    if len(found) >= 1:
        fi = normalize_date(found[0])
        ff = normalize_date(found[1]) if len(found) >= 2 else None
        return fi, ff

    fecha_pat = r"\b(\d{1,2}\s+de\s+[A-Za-záéíóúñ]+(?:\s+de)?\s+\d{4})\b"
    found2 = re.findall(fecha_pat, text, flags=re.IGNORECASE)
    if len(found2) >= 1:
        fi = normalize_date(found2[0])
        ff = normalize_date(found2[1]) if len(found2) >= 2 else None
        return fi, ff
    return None, None

def extract_preaviso(text: str) -> str:
    t = text.lower()
    m = re.search(r"(?:preaviso|antelación|avisar con)\s*(\d{1,4})\s*(día|días|mes|meses)", t)
    if m:
        num, unidad = m.group(1), m.group(2)
        return f"{num} {unidad}"
    m2 = re.search(r"(\d{1,4})\s*(día|días|mes|meses)\s+de\s+preaviso", t)
    if m2:
        num, unidad = m2.group(1), m2.group(2)
        return f"{num} {unidad}"
    return "NA"

# ==========================
# EXTRACCIÓN MEJORADA POR TEMA
# ==========================

def extract_desembolsos_pagos(paragraph: Optional[str]) -> Dict:
    """Mejorado para enfocarse en pagos extraordinarios, no en órdenes regulares"""
    if not paragraph:
        return {
            "TieneDesembolsos": "NO",
            "PeriodicidadPagos": "NA",
            "CondicionesPago": "No hay nada significativo.",
            "FormaPago": "Otro",
            "DetalleDesembolsos": "No hay nada significativo."
        }
    
    # Enfoque en pagos extraordinarios, excluyendo órdenes regulares
    tiene = find_bool_from_text(
        paragraph,
        positive_cues=[
            "bono de firma", "upfront", "bono de apertura", "pago inicial", 
            "desembolso extraordinario", "compensación adicional", "pago único",
            "incentivo de firma", "pago por suscripción", "desembolso de suma",
            "pago de bienvenida", "bono por apertura"
        ],
        negative_cues=[
            "no habrá pagos", "sin pagos", "no se realizan pagos", "solo órdenes",
            "únicamente órdenes", "excluyendo órdenes"
        ]
    )
    
    periodicidad = extract_periodicidad_pagos(paragraph)
    forma = extract_forma_pago(paragraph)
    
    # Mejorar extracción de condiciones preservando más contenido
    condiciones = pick_sentence_excerpt(paragraph, ["condiciones", "pago", "desembolso", "bono"], 800)
    detalle = paragraph.strip()
    
    return {
        "TieneDesembolsos": "SI" if tiene == "SI" else "NO",
        "PeriodicidadPagos": periodicidad,
        "CondicionesPago": text_or_na(condiciones),
        "FormaPago": forma,
        "DetalleDesembolsos": text_or_na(detalle)
    }

def extract_exclusividad(paragraph: Optional[str]) -> Dict:
    """Mejorado para detectar exclusividad de plataforma vs otras plataformas"""
    if not paragraph:
        return {
            "TieneExclusividad": "NO",
            "AlcanceExclusividad": "NA",
            "CondicionesExclusividad": "No hay nada significativo.",
            "RupturaExclusividad": "NO",
            "DetalleExclusividad": "No hay nada significativo."
        }
    
    t = paragraph.lower()
    
    # Mejor detección de exclusividad enfocada en plataformas
    tiene = "NO"
    if any(term in t for term in [
        "exclusividad", "acuerdo exclusivo", "no puede vender en otras plataformas",
        "prohibido usar otras plataformas", "exclusivo con rappi", "no competencia",
        "restricción de plataformas", "exclusividad de venta"
    ]):
        if not any(neg in t for neg in ["no exclusiv", "sin exclusiv"]) and not has_negation_near("exclusiv", paragraph):
            tiene = "SI"

    # Mejor detección del alcance
    alcance = "NA"
    if "territor" in t:
        alcance = "Territorial"
    elif any(x in t for x in ["producto", "servicio", "portafolio"]):
        alcance = "Producto/Servicio"
    elif "cliente" in t:
        alcance = "Clientes"
    elif any(x in t for x in ["plataforma", "uber eats", "didi food", "otras plataformas"]):
        alcance = "Plataforma"
    
    # Mejor detección de condiciones de ruptura
    ruptura = "NO"
    if any(x in t for x in [
        "romper", "terminar", "resolver", "rescindir", "incumplimiento", 
        "causales", "condiciones para terminar", "procedimiento de terminación"
    ]):
        ruptura = "SI"

    condiciones = pick_sentence_excerpt(paragraph, ["exclusiv", "alcance", "clientes", "territorio", "plataforma"], 800)
    detalle = paragraph.strip()
    
    return {
        "TieneExclusividad": tiene,
        "AlcanceExclusividad": alcance,
        "CondicionesExclusividad": text_or_na(condiciones),
        "RupturaExclusividad": ruptura,
        "DetalleExclusividad": text_or_na(detalle)
    }

def extract_termino(paragraph: Optional[str]) -> Dict:
    """Mejorado para enfocarse en terminación unilateral del aliado"""
    if not paragraph:
        return {
            "DuracionContrato": "NA",
            "FechaInicio": "",
            "FechaFin": "",
            "TerminacionUnilateral": "NO",
            "PenalidadTerminacion": "NO",
            "Preaviso": "NA",
            "DetalleTermino": "No hay nada significativo."
        }
    
    dur = extract_duracion(paragraph)
    fi, ff = extract_dates(paragraph)
    t = paragraph.lower()
    
    # Mejor detección de terminación unilateral
    terminacion_unilateral = "NO"
    if any(x in t for x in [
        "terminación unilateral", "dar por terminado", "terminar unilateralmente",
        "terminar el contrato", "rescindir el contrato", "terminación anticipada",
        "condiciones de terminación", "cómo terminar el contrato"
    ]):
        terminacion_unilateral = "SI"
    
    # Mejor detección de penalidades
    penalidad = "NO"
    if any(x in t for x in [
        "cláusula penal", "penalidad", "multa", "sanción", "devolución del bono",
        "penalidad por terminación", "compensación por terminación"
    ]):
        penalidad = "SI"
    
    preaviso = extract_preaviso(paragraph)
    detalle = paragraph.strip()
    
    return {
        "DuracionContrato": dur,
        "FechaInicio": fi or "",
        "FechaFin": ff or "",
        "TerminacionUnilateral": terminacion_unilateral,
        "PenalidadTerminacion": penalidad,
        "Preaviso": preaviso,
        "DetalleTermino": text_or_na(detalle)
    }

def extract_vigencia_renovacion(paragraph: Optional[str]) -> Dict:
    if not paragraph:
        return {
            "TieneRenovacionAutomatica": "NO",
            "PeriodicidadRenovacion": "NA",
            "PreavisoNoRenovacion": "NA",
            "DetalleVigencia": "No hay nada significativo."
        }
    
    t = paragraph.lower()
    tiene_auto = "NO"
    if ("renovación automática" in t or "renovacion automatica" in t) and not ("no " in t[: t.find("renov")] if "renov" in t else False):
        tiene_auto = "SI"

    periodicidad = "NA"
    if any(x in t for x in ["anual", "cada año", "cada 12 meses"]):
        periodicidad = "Anual"
    elif any(x in t for x in ["semestral", "cada 6 meses"]):
        periodicidad = "Semestral"
    elif "mensual" in t:
        periodicidad = "Mensual"
    else:
        if "renovación" in t or "renovacion" in t:
            periodicidad = "Otro"

    preaviso_no_ren = extract_preaviso(paragraph)
    detalle = paragraph.strip()
    
    return {
        "TieneRenovacionAutomatica": tiene_auto,
        "PeriodicidadRenovacion": periodicidad,
        "PreavisoNoRenovacion": preaviso_no_ren if preaviso_no_ren != "NA" else "NA",
        "DetalleVigencia": text_or_na(detalle)
    }

# ==========================
# MODO EMBEDDINGS
# ==========================

def process_file_embeddings(client_unused, path: Path) -> Dict:
    """Procesa un archivo y extrae información usando embeddings"""
    text = read_txt(path)
    chunks = paragraph_chunk(text)

    # Para cada tema, selecciona el mejor párrafo por embeddings
    best_paragraphs = {}
    for key, queries in THEME_QUERIES.items():
        best_p, best_sim = best_paragraph_for_theme(client_unused, chunks, queries, EMBEDDING_MODEL)
        best_paragraphs[key] = (best_p if best_sim >= SIM_THRESHOLD else None, best_sim)

    # Construye el JSON final
    desembolsos = extract_desembolsos_pagos(best_paragraphs["DesembolsosPagos"][0])
    exclusividad = extract_exclusividad(best_paragraphs["Exclusividad"][0])
    termino = extract_termino(best_paragraphs["TerminoContrato"][0])
    vigencia = extract_vigencia_renovacion(best_paragraphs["VigenciaRenovacion"][0])

    result = {
        "archivo": str(path.name),
        "DesembolsosPagos": desembolsos,
        "Exclusividad": exclusividad,
        "TerminoContrato": termino,
        "VigenciaRenovacion": vigencia,
        "_debug": {
            "similitudes": {
                "DesembolsosPagos": round(best_paragraphs["DesembolsosPagos"][1], 4),
                "Exclusividad": round(best_paragraphs["Exclusividad"][1], 4),
                "TerminoContrato": round(best_paragraphs["TerminoContrato"][1], 4),
                "VigenciaRenovacion": round(best_paragraphs["VigenciaRenovacion"][1], 4),
            },
            "umbral": SIM_THRESHOLD
        }
    }
    return result

def run_embeddings_mode():
    """Ejecuta el modo embeddings y guarda resultados en json_embeddings/"""
    print("🔍 Ejecutando modo embeddings...")
    
    # Verificar que Ollama esté disponible
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [model.get("name", "") for model in models]
        
        if OLLAMA_EMBED_MODEL not in model_names:
            print(f"ADVERTENCIA: El modelo {OLLAMA_EMBED_MODEL} no está disponible en Ollama.")
            print(f"Modelos disponibles: {model_names}")
            print(f"Para instalar el modelo, ejecuta: ollama pull {OLLAMA_EMBED_MODEL}")
            raise RuntimeError(f"Modelo {OLLAMA_EMBED_MODEL} no encontrado en Ollama")
        else:
            print(f"✓ Modelo {OLLAMA_EMBED_MODEL} encontrado en Ollama")
            
    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con Ollama en {OLLAMA_BASE_URL}: {e}")
        print("Asegúrate de que Ollama esté ejecutándose y sea accesible.")
        raise RuntimeError(f"No se puede conectar con Ollama: {e}")
    
    client_unused = None
    files = sorted([p for p in INPUT_DIR.glob("*.txt") if p.is_file()])
    if not files:
        print(f"No se encontraron .txt en {INPUT_DIR.resolve()}")
        return

    # Asegura carpeta de resultados
    OUTPUT_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    for path in files:
        try:
            result = process_file_embeddings(client_unused, path)
            # Guarda cada archivo individualmente
            output_file = OUTPUT_EMBEDDINGS_DIR / f"{path.stem}.json"
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✓ Procesado: {path.name}")
        except Exception as e:
            print(f"❌ Error en {path.name}: {e}")

# ==========================
# MODO RESUMEN
# ==========================

def generate_summary_with_ollama(text: str, field_name: str) -> str:
    """Genera un resumen usando Ollama LLM"""
    try:
        prompt = f"""Resume el siguiente texto extraído de un contrato comercial, enfocándote en los puntos clave y manteniendo la información importante. El texto es sobre: {field_name}

Texto a resumir:
{text}

Resumen (máximo 200 palabras):"""

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "max_tokens": 300
                }
            },
            timeout=120
        )
        response.raise_for_status()
        
        result = response.json()
        summary = result.get("response", "").strip()
        
        # Limpiar el resumen de posibles prefijos del modelo
        if summary.startswith("Resumen:"):
            summary = summary[8:].strip()
        if summary.startswith("El texto trata sobre"):
            summary = summary[20:].strip()
            
        return summary if summary else "No se pudo generar resumen."
        
    except Exception as e:
        print(f"Error generando resumen para {field_name}: {e}")
        return "Error al generar resumen."

def process_file_with_summary(input_file: Path) -> Dict:
    """Procesa un archivo JSON de embeddings y reemplaza campos de detalle con resúmenes"""
    try:
        with input_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Campos que necesitan resumen (reemplazar contenido)
        fields_to_summarize = [
            ("DesembolsosPagos", "DetalleDesembolsos"),
            ("Exclusividad", "CondicionesExclusividad"),
            ("Exclusividad", "DetalleExclusividad"),
            ("TerminoContrato", "DetalleTermino"),
            ("VigenciaRenovacion", "DetalleVigencia")
        ]
        
        # Generar resúmenes y reemplazar contenido
        for section, field in fields_to_summarize:
            if section in data and field in data[section]:
                original_text = data[section][field]
                if original_text and original_text != "No hay nada significativo.":
                    summary = generate_summary_with_ollama(original_text, f"{section}.{field}")
                    # Reemplazar el contenido original con el resumen
                    data[section][field] = summary
        
        return data
        
    except Exception as e:
        print(f"Error procesando {input_file.name}: {e}")
        return None

def run_summary_mode():
    """Ejecuta el modo resumen procesando archivos de json_embeddings/"""
    print("📝 Ejecutando modo resumen...")
    print(f"🤖 Modelo LLM para resúmenes: {OLLAMA_LLM_MODEL}")
    
    # Verificar que Ollama esté disponible
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [model.get("name", "") for model in models]
        
        if OLLAMA_LLM_MODEL not in model_names:
            print(f"ADVERTENCIA: El modelo {OLLAMA_LLM_MODEL} no está disponible en Ollama.")
            print(f"Modelos disponibles: {model_names}")
            print(f"Para instalar el modelo, ejecuta: ollama pull {OLLAMA_LLM_MODEL}")
            raise RuntimeError(f"Modelo {OLLAMA_LLM_MODEL} no encontrado en Ollama")
        else:
            print(f"✓ Modelo {OLLAMA_LLM_MODEL} encontrado en Ollama")
            
    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con Ollama en {OLLAMA_BASE_URL}: {e}")
        print("Asegúrate de que Ollama esté ejecutándose y sea accesible.")
        raise RuntimeError(f"No se puede conectar con Ollama: {e}")
    
    # Buscar archivos JSON en json_embeddings/
    json_files = sorted([p for p in OUTPUT_EMBEDDINGS_DIR.glob("*.json") if p.is_file()])
    if not json_files:
        print(f"No se encontraron archivos JSON en {OUTPUT_EMBEDDINGS_DIR.resolve()}")
        print("Ejecuta primero el modo embeddings con: python main.py --embeddings")
        return

    # Asegura carpeta de resultados
    OUTPUT_RESUME_DIR.mkdir(parents=True, exist_ok=True)

    for json_file in json_files:
        try:
            result = process_file_with_summary(json_file)
            if result:
                # Guarda el archivo con resúmenes
                output_file = OUTPUT_RESUME_DIR / f"{json_file.stem}_with_resume.json"
                with output_file.open("w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"✓ Procesado con resúmenes: {json_file.name}")
        except Exception as e:
            print(f"❌ Error procesando {json_file.name}: {e}")

# ==========================
# MAIN
# ==========================

def main():
    parser = argparse.ArgumentParser(description="Extractor semántico de cláusulas de contratos")
    parser.add_argument("--embeddings", action="store_true", 
                       help="Ejecutar modo embeddings (extrae información usando embeddings)")
    parser.add_argument("--summary", action="store_true", 
                       help="Ejecutar modo resumen (genera resúmenes de campos de detalle)")
    parser.add_argument("--llm-model", type=str, default=None,
                       help="Modelo LLM para generar resúmenes (ej: llama3.1, gemma3, qwen3:8b)")
    
    args = parser.parse_args()
    
    if not args.embeddings and not args.summary:
        print("❌ Debes especificar --embeddings o --summary")
        print("Uso:")
        print("  python main.py --embeddings    # Extrae información usando embeddings")
        print("  python main.py --summary       # Genera resúmenes de campos de detalle")
        print("  python main.py --summary --llm-model llama3.1  # Usar modelo específico para resúmenes")
        print("\nModelos disponibles en tu sistema:")
        print("  - llama3.1")
        print("  - gemma3")
        print("  - qwen3:8b")
        print("  - gemma3:4b")
        print("  - orca-mini")
        return
    
    # Si se especifica un modelo LLM, actualizar la variable global
    if args.llm_model:
        global OLLAMA_LLM_MODEL
        OLLAMA_LLM_MODEL = args.llm_model
        print(f"🔧 Usando modelo LLM: {OLLAMA_LLM_MODEL}")
    
    if args.embeddings:
        run_embeddings_mode()
    
    if args.summary:
        run_summary_mode()

if __name__ == "__main__":
    main()