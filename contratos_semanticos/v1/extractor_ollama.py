# -*- coding: utf-8 -*-
"""
Extractor semántico de cláusulas de contratos (.txt) -> JSON por archivo.

Motor semántico: Ollama Embeddings (nomic-embed-text por defecto).
Docs oficiales (Ollama): https://ollama.com/library/nomic-embed-text
"""

import os
import re
import json
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

INPUT_DIR = Path("input_txt")
OUTPUT_JSONL = Path("salida_contratos.jsonl")
OUTPUT_JSON = Path("salida_contratos.json")
OUTPUT_DIR_PER_FILE = Path("jsonResults")

load_dotenv()
# Configuración de Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL")

# Modelo embeddings (usando Ollama)
EMBEDDING_MODEL = OLLAMA_EMBED_MODEL

# Umbral de similitud para considerar un párrafo "relevante"
SIM_THRESHOLD = 0.45

# Límite conservador de caracteres por chunk para evitar exceder el contexto
# de entrada del modelo de embeddings. Aproximación: ~4 caracteres por token.
# Mantener los chunks por debajo de ~2000-3000 tokens por seguridad.
MAX_CHARS_PER_CHUNK = 8000

# Batch para reducir llamadas (embeddeamos varios párrafos/input a la vez)
BATCH_SIZE = 64

# Prompts/keywords por tema (en español, orientados a contratos)
THEME_QUERIES = {
    "DesembolsosPagos": [
        "desembolso", "desembolsos", "pago", "pagos", "condiciones de pago",
        "periodicidad de pagos", "forma de pago", "transferencia", "efectivo",
        "contra hitos", "cronograma de pagos", "abonos", "fechas de pago"
    ],
    "Exclusividad": [
        "exclusividad", "acuerdo exclusivo", "no exclusividad", "exclusivo",
        "territorio exclusivo", "clientes exclusivos", "alcance de la exclusividad",
        "ruptura de exclusividad", "causales para terminar la exclusividad"
    ],
    "TerminoContrato": [
        "término del contrato", "duración del contrato", "plazo del contrato",
        "terminación unilateral", "cláusula penal", "penalidad por terminación",
        "preaviso", "fecha de inicio", "fecha de terminación", "fecha fin"
    ],
    "VigenciaRenovacion": [
        "vigencia", "renovación", "renovación automática", "no renovación",
        "preaviso de no renovación", "periodicidad de renovación", "término y renovación"
    ]
}

# ==========================
# UTILIDADES
# ==========================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # Asume vectores normalizados; si no, normaliza
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
    # Separación simple por oraciones (heurística basada en puntuación)
    sentences = re.split(r"(?<=[\.!?])\s+", text)
    # Normaliza espacios residuales
    return [re.sub(r"[ \t]+", " ", s).strip() for s in sentences if s and s.strip()]

def _safe_chunk_long_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    # Intenta dividir por oraciones manteniendo el límite de caracteres
    sentences = _split_by_sentences(text)
    parts: List[str] = []
    current = []
    current_len = 0
    for s in sentences:
        s_len = len(s) + 1  # espacio
        if current_len + s_len <= max_chars or not current:
            current.append(s)
            current_len += s_len
        else:
            parts.append(" ".join(current).strip())
            current = [s]
            current_len = s_len
    if current:
        parts.append(" ".join(current).strip())
    # Si aún quedan fragmentos desproporcionados, divide por longitud fija
    final_parts: List[str] = []
    for part in parts:
        if len(part) <= max_chars:
            final_parts.append(part)
        else:
            for i in range(0, len(part), max_chars):
                final_parts.append(part[i:i+max_chars])
    return final_parts

def paragraph_chunk(text: str) -> List[str]:
    # Divide por doble salto de línea; limpia viñetas y espacios
    raw = re.split(r"\n{2,}", text)
    chunks: List[str] = []
    for p in raw:
        pp = re.sub(r"[ \t]+", " ", p).strip()
        pp = re.sub(r"^[•\-\u2022]+\s*", "", pp)  # quita viñetas iniciales
        if len(pp) >= 20:
            # Garantiza que ningún chunk exceda el límite conservador
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

def pick_sentence_excerpt(text: str, keys: List[str], fallback_chars: int = 320) -> str:
    # Devuelve la oración más alineada a las keys o un recorte del párrafo
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
    return text.strip()[:fallback_chars].strip()

# ------------------ Extracciones específicas ------------------

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
        "contra hitos", "por hitos", "a la entrega"
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

# ------------------ Ollama Embeddings helpers ------------------

def _ollama_embed_batch(texts: List[str], model: str) -> np.ndarray:
    """
    Ollama /api/embeddings no admite batch en un solo request; hacemos loop.
    Implementa manejo de errores y reintentos para mayor robustez.
    """
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
                break  # Éxito, salir del loop de reintentos
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:  # Último intento
                    print(f"Error al obtener embedding para texto (intento {attempt + 1}): {e}")
                    # Agregar un vector cero como fallback
                    embs.append(np.zeros(768, dtype=np.float32))  # nomic-embed-text tiene 768 dimensiones
                else:
                    print(f"Reintentando embedding (intento {attempt + 1}/{max_retries}): {e}")
                    continue
    return np.vstack(embs) if embs else np.zeros((0, 0), dtype=np.float32)

def embed_texts(client_unused, texts: List[str], model: str) -> np.ndarray:
    """
    Embebe una lista de textos usando Ollama, retorna matriz (N x D).
    El parámetro client_unused se mantiene para compatibilidad con la interfaz existente.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    return _ollama_embed_batch(texts, OLLAMA_EMBED_MODEL)

def embed_centroid(client_unused, queries: List[str], model: str) -> np.ndarray:
    """
    Calcula un "centroide" (promedio) de embeddings para un conjunto de queries usando Ollama.
    Esto funciona bien para definir la intención semántica del tema.
    El parámetro client_unused se mantiene para compatibilidad con la interfaz existente.
    """
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
    # Centroide del tema
    theme_vec = embed_centroid(client_unused, theme_queries, model)
    # Embeddings de párrafos
    para_embs = embed_texts(client_unused, chunks, model)
    if para_embs.size == 0:
        return None, 0.0
    # Normaliza
    norms = np.linalg.norm(para_embs, axis=1, keepdims=True)
    para_embs = para_embs / np.clip(norms, 1e-8, None)
    # Similaridades
    sims = para_embs @ theme_vec.reshape(-1, 1)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx, 0])
    return chunks[best_idx], best_sim

# ==========================
# EXTRACCIÓN POR TEMA -> JSON
# ==========================

def extract_desembolsos_pagos(paragraph: Optional[str]) -> Dict:
    if not paragraph:
        return {
            "TieneDesembolsos": "NO",
            "PeriodicidadPagos": "NA",
            "CondicionesPago": "No hay nada significativo.",
            "FormaPago": "Otro",
            "DetalleDesembolsos": "No hay nada significativo."
        }
    tiene = find_bool_from_text(
        paragraph,
        positive_cues=["desembolso", "pago", "pagos", "cronograma", "abono", "transferencia"],
        negative_cues=["no habrá pagos", "sin pagos", "no se realizan pagos"]
    )
    periodicidad = extract_periodicidad_pagos(paragraph)
    forma = extract_forma_pago(paragraph)
    condiciones = pick_sentence_excerpt(paragraph, ["condiciones", "pago", "desembolso"])
    detalle = paragraph.strip()
    return {
        "TieneDesembolsos": "SI" if tiene == "SI" else "NO",
        "PeriodicidadPagos": periodicidad,
        "CondicionesPago": text_or_na(condiciones),
        "FormaPago": forma,
        "DetalleDesembolsos": text_or_na(detalle)
    }

def extract_exclusividad(paragraph: Optional[str]) -> Dict:
    if not paragraph:
        return {
            "TieneExclusividad": "NO",
            "AlcanceExclusividad": "NA",
            "CondicionesExclusividad": "No hay nada significativo.",
            "RupturaExclusividad": "NO",
            "DetalleExclusividad": "No hay nada significativo."
        }
    t = paragraph.lower()
    if "exclusiv" in t:
        if "no exclusiv" in t or "sin exclusiv" in t or has_negation_near("exclusiv", paragraph):
            tiene = "NO"
        else:
            tiene = "SI"
    else:
        tiene = "NO"

    alcance = "NA"
    if "territor" in t:
        alcance = "Territorial"
    if any(x in t for x in ["producto", "servicio", "portafolio"]):
        alcance = "Producto/Servicio" if alcance == "NA" else alcance
    if "cliente" in t:
        alcance = "Clientes" if alcance == "NA" else alcance

    ruptura = "SI" if any(x in t for x in ["romper", "terminar", "resolver", "rescindir", "incumplimiento", "causales"]) else "NO"

    condiciones = pick_sentence_excerpt(paragraph, ["exclusiv", "alcance", "clientes", "territorio"])
    detalle = paragraph.strip()
    return {
        "TieneExclusividad": tiene,
        "AlcanceExclusividad": alcance,
        "CondicionesExclusividad": text_or_na(condiciones),
        "RupturaExclusividad": ruptura,
        "DetalleExclusividad": text_or_na(detalle)
    }

def extract_termino(paragraph: Optional[str]) -> Dict:
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
    terminacion_unilateral = "SI" if any(
        x in t for x in ["terminación unilateral", "dar por terminado", "terminar unilateralmente"]
    ) else "NO"
    penalidad = "SI" if any(
        x in t for x in ["cláusula penal", "penalidad", "multa", "sanción"]
    ) else "NO"
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
    tiene_auto = "SI" if ("renovación automática" in t or "renovacion automatica" in t) and not ("no " in t[: t.find("renov")] if "renov" in t else False) else "NO"

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
# PIPELINE
# ==========================

def process_file(client_unused, path: Path) -> Dict:
    text = read_txt(path)
    chunks = paragraph_chunk(text)

    # Para cada tema, selecciona el mejor párrafo por embeddings
    best_paragraphs = {}
    for key, queries in THEME_QUERIES.items():
        best_p, best_sim = best_paragraph_for_theme(client_unused, chunks, queries, EMBEDDING_MODEL)
        best_paragraphs[key] = (best_p if best_sim >= SIM_THRESHOLD else None, best_sim)

    # Construye el JSON final (cada padre es una propiedad del objeto raíz)
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

def main():
    load_dotenv()  # permite usar .env
    
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
    
    # client_unused se mantiene para compatibilidad con la interfaz existente
    client_unused = None

    files = sorted([p for p in INPUT_DIR.glob("*.txt") if p.is_file()])
    if not files:
        print(f"No se encontraron .txt en {INPUT_DIR.resolve()}")
        return

    # Asegura carpeta de resultados por archivo
    OUTPUT_DIR_PER_FILE.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for path in files:
            try:
                result = process_file(client_unused, path)
                results.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                # Guarda también un JSON formateado por archivo en jsonResults/
                per_file_json = OUTPUT_DIR_PER_FILE / f"{path.stem}.result.json"
                with per_file_json.open("w", encoding="utf-8") as pf:
                    json.dump(result, pf, ensure_ascii=False, indent=2)
                print(f"OK -> {path.name}")
            except Exception as e:
                print(f"ERROR en {path.name}: {e}")

    # Además del JSONL, genera un JSON "limpio" y formateado (para kller)
    try:
        # Si hay un solo resultado, guarda el objeto; si hay varios, guarda una lista
        pretty_payload = results[0] if len(results) == 1 else results
        with OUTPUT_JSON.open("w", encoding="utf-8") as jf:
            json.dump(pretty_payload, jf, ensure_ascii=False, indent=2)
        print(f"JSON formateado -> {OUTPUT_JSON.name}")
    except Exception as e:
        print(f"ERROR al escribir {OUTPUT_JSON.name}: {e}")

if __name__ == "__main__":
    main()