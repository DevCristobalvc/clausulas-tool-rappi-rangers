# -*- coding: utf-8 -*-
"""
Pipeline híbrido por campo (solo Ollama) para contratos en outputs/*.txt

- Retrieval con embeddings de Ollama (top-k, MMR)
- LLM de Ollama (llama3.1) para booleans con matiz legal y resúmenes breves
- Regex/reglas para fechas, duraciones, preavisos, periodicidades y formas de pago
- Genera json_ollama/*.json con el esquema solicitado

Requisitos:
  pip install numpy requests dateparser

Variables de entorno recomendadas:
  OLLAMA_BASE_URL=http://localhost:11434
  OLLAMA_LLM_MODEL=llama3.1:latest
  OLLAMA_EMBED_MODEL=paraphrase-multilingual:278m-mpnet-base-v2-fp16
"""

import os, re, json, unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import requests
import dateparser

# ======================
# CONFIG
# ======================
INPUT_DIR = Path("outputs")
OUTPUT_DIR = Path("json_ollama")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:latest")

# Embeddings multilingües (buen recall en español).
# Alternativas según tu "llama list": embeddinggemma:latest, mxbai-embed-large:latest,
# nomic-embed-text:latest. Cambia aquí o vía variable de entorno.
OLLAMA_EMBED_MODEL = os.getenv(
    "OLLAMA_EMBED_MODEL",
    "paraphrase-multilingual:278m-mpnet-base-v2-fp16"
)

TOP_K = 5          # cuántos fragmentos usar por tema
SIM_THRESHOLD = 0.40
MMR_LAMBDA = 0.70  # 1.0 = solo relevancia, 0.0 = solo diversidad
MAX_CTX_CHARS = 8000

# Consultas por tema (para el vector de consulta)
THEME_QUERIES = {
    "DesembolsosPagos": [
        "bono de firma", "upfront", "pago inicial", "desembolso extraordinario",
        "pago único", "incentivo de firma", "compensación adicional", "bono por apertura"
    ],
    "Exclusividad": [
        "exclusividad", "acuerdo exclusivo", "no competencia", "otras plataformas",
        "exclusivo con", "restricción de plataformas", "exclusividad de venta",
        "exclusividad territorial", "exclusividad de producto", "clientes exclusivos"
    ],
    "TerminoContrato": [
        "terminación unilateral", "dar por terminado", "rescindir el contrato",
        "terminación anticipada", "preaviso", "cláusula penal", "multa", "sanción",
        "vigencia del contrato", "plazo del contrato"
    ],
    "VigenciaRenovacion": [
        "renovación automática", "prórroga", "extensión del contrato",
        "periodicidad de renovación", "preaviso de no renovación", "vigencia"
    ],
}

# ======================
# UTILS
# ======================
def norm(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def split_segments(text: str) -> List[Dict]:
    """
    Segmenta priorizando posibles títulos de cláusula; si no, por dobles saltos.
    Devuelve dicts con: {"text", "offset"}
    Mejora: detecta encabezados tipo "DÉCIMA OCTAVA. Vigencia..." para agrupar la cláusula completa.
    """
    text = text.replace("\r\n", "\n")
    parts = []
    # split por posibles encabezados de cláusula (incluye ordinales "DÉCIMA", "DÉCIMO", etc.)
    pattern = re.compile(
        r"(?im)^(?:cl[áa]usula\s+\d+|d[ée]cim[ao](?:\s+\w+)?\.|vigencia(?:\s+y\s+terminaci[oó]n)?|t[ée]rminos?|pagos?|exclusividad)\b.*$"
    )
    indices = [m.start() for m in pattern.finditer(text)]
    if indices:
        indices = sorted(set([0] + indices + [len(text)]))
        for i in range(len(indices)-1):
            seg = text[indices[i]:indices[i+1]]
            seg = seg.strip()
            if len(seg) > 40:
                parts.append({"text": seg, "offset": indices[i]})
    else:
        # fallback: doble salto
        for m in re.split(r"\n{2,}", text):
            seg = norm(m)
            if len(seg) > 40:
                parts.append({"text": seg, "offset": text.find(m)})
    return parts

def call_ollama_embeddings(texts: List[str]) -> np.ndarray:
    embs = []
    for t in texts:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": t},
            timeout=120
        )
        r.raise_for_status()
        embs.append(np.array(r.json()["embedding"], dtype=np.float32))
    return np.vstack(embs)

def l2norm(x: np.ndarray) -> np.ndarray:
    denom = np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
    return x / denom

def embed_centroid(queries: List[str]) -> np.ndarray:
    v = call_ollama_embeddings(queries)
    v = v.mean(axis=0, keepdims=True)
    return l2norm(v)[0]

def cosine_sim_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    # A: (n,d) ya normalizado; b: (d,)
    return (A @ b.reshape(-1, 1)).ravel()

def mmr_select(texts: List[str], embs: np.ndarray, q: np.ndarray, k: int, lamb: float = 0.7) -> List[int]:
    """
    Maximal Marginal Relevance para diversidad + relevancia.
    """
    embs = l2norm(embs)
    q = q / max(np.linalg.norm(q), 1e-8)
    sim_to_q = cosine_sim_matrix(embs, q)
    selected = []
    candidates = list(range(len(texts)))
    while candidates and len(selected) < k:
        if not selected:
            # elegir el más similar entre los candidatos (no por índice posicional)
            best_local_idx = int(np.argmax([sim_to_q[c] for c in candidates]))
            selected.append(candidates.pop(best_local_idx))
        else:
            best_i = None
            best_score = -1e9
            for idx in candidates:
                max_sim_sel = max(embs[idx] @ embs[j] for j in selected)
                score = lamb * sim_to_q[idx] - (1 - lamb) * max_sim_sel
                if score > best_score:
                    best_score, best_i = score, idx
            selected.append(best_i)
            candidates.remove(best_i)
    return selected

def build_context(segments: List[Dict], max_chars: int = MAX_CTX_CHARS) -> str:
    ctx = "\n\n---\n\n".join(norm(s["text"]) for s in segments)
    return ctx[:max_chars]

def call_ollama_json(prompt: str, model: str = None) -> Optional[dict]:
    if model is None: model = OLLAMA_LLM_MODEL
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0}
            },
            timeout=180
        )
        r.raise_for_status()
        data = r.json().get("response", "").strip()
        # limpia posibles fences accidentales
        data = data.strip().lstrip("```json").lstrip("```").strip()
        return json.loads(data)
    except Exception as e:
        print(f"[LLM] Error JSON: {e}")
        return None

# -------- NUEVO: devolver el texto COMPLETO de los segmentos (sin cortes) --------
def full_from_segments(segments: List[Dict]) -> str:
    """
    Devuelve el texto COMPLETO de los segmentos seleccionados, sin cortar.
    Separa cada segmento con una línea '---' para legibilidad.
    """
    if not segments:
        return "No hay nada significativo."
    return "\n\n---\n\n".join(norm(s["text"]) for s in segments)

# ======================
# REGLAS / REGEX
# ======================
def extract_periodicidad(text: str) -> str:
    t = text.lower()
    if "mensual" in t: return "Mensual"
    if "quincenal" in t: return "Quincenal"
    if "semanal" in t: return "Semanal"
    if "bimensual" in t: return "Bimensual"
    if "trimestral" in t: return "Trimestral"
    if "por hitos" in t or "contra hitos" in t or "a la entrega" in t: return "Contra Hitos"
    if "una sola vez" in t or "único" in t: return "Único"
    return "NA"

def extract_forma_pago(text: str) -> str:
    t = text.lower()
    if "transferencia" in t or "depósito" in t or "deposito" in t: return "Transferencia"
    if "efectivo" in t: return "Efectivo"
    if "cheque" in t: return "Cheque"
    if "pse" in t: return "PSE"
    if "tarjeta" in t: return "Otro"
    return "Otro"

def extract_duracion(text: str) -> str:
    t = text.lower()
    m = re.search(r"(\d{1,3})\s*(mes|meses|año|años)", t)
    if m:
        n, u = m.group(1), m.group(2)
        u = "meses" if "mes" in u else "años"
        return f"{n} {u}"
    if "indefinid" in t: return "Indefinido"
    return "NA"

def parse_date_any(s: str) -> Optional[str]:
    dt = dateparser.parse(s, languages=["es", "en"])
    return dt.strftime("%Y-%m-%d") if dt else None

def extract_dates(text: str) -> Tuple[str, str]:
    fi, ff = "", ""
    iso = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if iso:
        fi = parse_date_any(iso[0]) or ""
        if len(iso) > 1:
            ff = parse_date_any(iso[1]) or ""
        return fi, ff
    f = re.findall(r"\b(\d{1,2}\s+de\s+[A-Za-záéíóúñ]+(?:\s+de)?\s+\d{4})\b", text, flags=re.I)
    if f:
        fi = parse_date_any(f[0]) or ""
        if len(f) > 1:
            ff = parse_date_any(f[1]) or ""
    return fi, ff

def extract_preaviso(text: str) -> str:
    t = text.lower()
    m = re.search(r"(?:preaviso|antelaci[oó]n|avisar con)\s*(\d{1,4})\s*(d[ií]a[s]?|mes(?:es)?)", t, flags=re.I)
    if not m:
        m = re.search(r"(\d{1,4})\s*(d[ií]a[s]?|mes(?:es)?)\s+de\s+preaviso", t, flags=re.I)
    if m:
        n, u = m.group(1), m.group(2)
        u = "días" if u.lower().startswith("d") else "meses"
        return f"{n} {u}"
    return "NA"

def excerpt_from_segments(segments: List[Dict], keywords: List[str], fallback_chars: int = 500) -> str:
    text = "\n ".join(s["text"] for s in segments)
    low = text.lower()
    for kw in keywords:
        i = low.find(kw.lower())
        if i != -1:
            a = max(0, i - 350); b = min(len(text), i + len(kw) + 350)
            return norm(text[a:b])
    return norm(text[:fallback_chars]) if text else "No hay nada significativo."

def has_exclusividad_negada(text: str) -> bool:
    t = text.lower()
    return ("sin exclusiv" in t) or ("no exclusiv" in t)

# ======================
# EXTRACTORES POR CAMPO (LLM + reglas)
# ======================
def extractor_desembolsos(segments: List[Dict]) -> Dict:
    ctx = build_context(segments)
    # 1) LLM clasifica si hay desembolsos extraordinarios y resume condiciones
    prompt = f"""
Eres un analista legal. Con base SOLO en los fragmentos abajo (contratos en español), decide si existen DESEMBOLSOS o PAGOS EXTRAORDINARIOS (p.ej., bono de firma, upfront, pago único, incentivos; NO confundir con órdenes regulares).
Responde SOLO en JSON estricto con estas claves y valores permitidos:

{{
  "TieneDesembolsos": "SI" | "NO",
  "CondicionesPago": "texto corto (máx 2 frases o 'NA')"
}}

Fragmentos:
\"\"\"{ctx}\"\"\""""
    llm = call_ollama_json(prompt)
    tiene = (llm or {}).get("TieneDesembolsos", "NO")
    condiciones_llm = (llm or {}).get("CondicionesPago", "NA")

    # 2) Reglas deterministas
    flat = "\n ".join(s["text"] for s in segments).lower()
    pos = ["bono de firma","upfront","pago inicial","desembolso extraordinario","pago único",
           "incentivo de firma","compensación adicional","bono por apertura"]
    if tiene == "NO" and any(k in flat for k in pos):
        tiene = "SI"

    periodicidad = extract_periodicidad(flat) if tiene == "SI" else "NA"
    forma = extract_forma_pago(flat) if tiene == "SI" else "NA"

    condiciones = condiciones_llm if tiene == "SI" and condiciones_llm and condiciones_llm != "NA" \
        else excerpt_from_segments(segments, ["pago","bono","upfront","desembolso"])

    # *** Detalle COMPLETO sin cortes ***
    detalle = full_from_segments(segments) if tiene == "SI" else "No hay nada significativo."

    return {
        "TieneDesembolsos": "SI" if tiene == "SI" else "NO",
        "PeriodicidadPagos": periodicidad,
        "CondicionesPago": condiciones if tiene == "SI" else "No hay nada significativo.",
        "FormaPago": forma,
        "DetalleDesembolsos": detalle
    }

def extractor_exclusividad(segments: List[Dict]) -> Dict:
    ctx = build_context(segments)
    prompt = f"""
Eres un analista legal. Con base SOLO en los fragmentos, determina EXCLUSIVIDAD.
Responde SOLO en JSON estricto:

{{
  "TieneExclusividad": "SI" | "NO",
  "AlcanceExclusividad": "Territorial" | "Producto/Servicio" | "Clientes" | "Plataforma" | "Otro" | "NA",
  "RupturaExclusividad": "SI" | "NO",
  "CondicionesExclusividad": "texto corto (máx 2 frases o 'NA')"
}}

Reglas:
- Si se menciona exclusividad y no está negada, marca "SI".
- Plataforma = restricción sobre otras aplicaciones/plataformas (Uber Eats, etc.).
- Si no hay señales, usar "NO" y "NA".

Fragmentos:
\"\"\"{ctx}\"\"\""""
    llm = call_ollama_json(prompt) or {}
    tiene = llm.get("TieneExclusividad", "NO")
    alcance = llm.get("AlcanceExclusividad", "NA")
    ruptura = llm.get("RupturaExclusividad", "NO")
    condiciones = llm.get("CondicionesExclusividad", "NA")

    # Reglas de salvaguarda
    flat = "\n ".join(s["text"] for s in segments).lower()
    if has_exclusividad_negada(flat):
        tiene, alcance, ruptura, condiciones = "NO", "NA", "NO", "No hay nada significativo."
    if tiene == "SI":
        if "territor" in flat: alcance = "Territorial"
        elif any(x in flat for x in ["plataforma","uber eats","didi food","otras plataformas"]): alcance = "Plataforma"
        elif "cliente" in flat: alcance = "Clientes"
        elif any(x in flat for x in ["producto","servicio","portafolio"]): alcance = "Producto/Servicio"
        else: alcance = alcance if alcance in {"Territorial","Producto/Servicio","Clientes","Plataforma","Otro"} else "Otro"
        if any(x in flat for x in ["romper","terminar","rescindir","resolver","incumplimiento","causales"]):
            ruptura = "SI"

    # *** Detalle COMPLETO sin cortes ***
    detalle = full_from_segments(segments) if tiene == "SI" else "No hay nada significativo."

    return {
        "TieneExclusividad": tiene,
        "AlcanceExclusividad": alcance if tiene == "SI" else "NA",
        "CondicionesExclusividad": condiciones if tiene == "SI" else "No hay nada significativo.",
        "RupturaExclusividad": ruptura if tiene == "SI" else "NO",
        "DetalleExclusividad": detalle
    }

def extractor_termino(segments: List[Dict]) -> Dict:
    ctx = build_context(segments)
    prompt = f"""
Eres un analista legal. Con base SOLO en los fragmentos, indica si existe TERMINACIÓN UNILATERAL por cualquiera de las partes.
Responde SOLO en JSON estricto:

{{
  "TerminacionUnilateral": "SI" | "NO"
}}

Fragmentos:
\"\"\"{ctx}\"\"\""""
    llm = call_ollama_json(prompt) or {}
    term_uni = llm.get("TerminacionUnilateral", "NO")

    flat = "\n ".join(s["text"] for s in segments)
    dur = extract_duracion(flat)
    fi, ff = extract_dates(flat)
    penal = "SI" if re.search(r"cl[áa]usula penal|penalidad|multa|sanci[oó]n", flat, flags=re.I) else "NO"
    preav = extract_preaviso(flat)

    # *** Detalle COMPLETO sin cortes ***
    detalle = full_from_segments(segments)

    return {
        "DuracionContrato": dur,
        "FechaInicio": fi,
        "FechaFin": ff,
        "TerminacionUnilateral": term_uni,
        "PenalidadTerminacion": penal,
        "Preaviso": preav,
        "DetalleTermino": detalle if detalle else "No hay nada significativo."
    }

def extractor_vigencia(segments: List[Dict]) -> Dict:
    ctx = build_context(segments)
    prompt = f"""
Eres un analista legal. Con base SOLO en los fragmentos, decide si existe RENOVACIÓN AUTOMÁTICA.
Responde SOLO en JSON estricto:

{{
  "TieneRenovacionAutomatica": "SI" | "NO"
}}

Notas: si el texto dice que NO hay renovación automática, responde "NO".

Fragmentos:
\"\"\"{ctx}\"\"\""""
    llm = call_ollama_json(prompt) or {}
    tiene_auto = llm.get("TieneRenovacionAutomatica", "NO")

    flat = "\n ".join(s["text"] for s in segments)
    flat_norm = norm(flat).lower()

    # Mejora: detectar "se prorrogará/renovará automáticamente"
    auto_hits = [
        "renovación automática", "renovacion automatica",
        "se renovará automáticamente", "se renovara automaticamente",
        "se prorrogará automáticamente", "se prorrogara automaticamente",
        "prórroga automática", "prorroga automatica",
        "prorrogará automáticamente", "prorrogara automaticamente"
    ]
    neg_hits = [
        "sin renovación automática", "sin renovacion automatica",
        "no habrá renovación automática", "no habra renovacion automatica",
        "no aplica renovación automática", "no aplica renovacion automatica",
        "no renovación automática", "no renovacion automatica"
    ]
    if any(p in flat_norm for p in auto_hits):
        tiene_auto = "SI"
    if any(n in flat_norm for n in neg_hits):
        tiene_auto = "NO"

    per = "NA"
    if any(x in flat_norm for x in ["anual","cada 12 meses","cada año"]): per = "Anual"
    elif any(x in flat_norm for x in ["semestral","cada 6 meses"]): per = "Semestral"
    elif "mensual" in flat_norm: per = "Mensual"
    elif "renov" in flat_norm or "prórrog" in flat_norm or "prorrog" in flat_norm: per = "Otro"

    pre_no_ren = extract_preaviso(flat) if tiene_auto == "SI" else "NA"

    # *** Detalle COMPLETO sin cortes ***
    detalle = full_from_segments(segments) if ("renov" in flat_norm or "vigencia" in flat_norm or "prorrog" in flat_norm) else "No hay nada significativo."

    return {
        "TieneRenovacionAutomatica": tiene_auto,
        "PeriodicidadRenovacion": per if tiene_auto == "SI" else "NA",
        "PreavisoNoRenovacion": pre_no_ren,
        "DetalleVigencia": detalle
    }

# ======================
# RETRIEVAL POR TEMA
# ======================
def retrieve_for_theme(all_segments: List[Dict], queries: List[str], top_k: int = TOP_K) -> List[Dict]:
    texts = [s["text"] for s in all_segments]
    if not texts:
        return []
    seg_embs = call_ollama_embeddings(texts)
    q_vec = embed_centroid(queries)
    idxs = mmr_select(texts, seg_embs, q_vec, k=min(top_k, len(texts)), lamb=MMR_LAMBDA)
    # filtra por similitud mínima
    sims = cosine_sim_matrix(l2norm(seg_embs), q_vec)
    chosen = [i for i in idxs if sims[i] >= SIM_THRESHOLD]
    if not chosen:  # si todos están por debajo, al menos toma el top-1 absoluto
        chosen = [int(np.argmax(sims))]
    return [all_segments[i] for i in chosen]

# ======================
# ORQUESTACIÓN
# ======================
def process_contract_text(text: str) -> Dict:
    segments = split_segments(text)

    # Recupera top-k por tema
    seg_des = retrieve_for_theme(segments, THEME_QUERIES["DesembolsosPagos"], TOP_K)
    seg_exc = retrieve_for_theme(segments, THEME_QUERIES["Exclusividad"], TOP_K)
    seg_ter = retrieve_for_theme(segments, THEME_QUERIES["TerminoContrato"], TOP_K)
    seg_vig = retrieve_for_theme(segments, THEME_QUERIES["VigenciaRenovacion"], TOP_K)

    # Extrae por campo (LLM + reglas)
    out_des = extractor_desembolsos(seg_des) if seg_des else {
        "TieneDesembolsos": "NO", "PeriodicidadPagos": "NA",
        "CondicionesPago": "No hay nada significativo.", "FormaPago": "NA",
        "DetalleDesembolsos": "No hay nada significativo."
    }
    out_exc = extractor_exclusividad(seg_exc) if seg_exc else {
        "TieneExclusividad": "NO","AlcanceExclusividad": "NA","CondicionesExclusividad": "No hay nada significativo.",
        "RupturaExclusividad": "NO","DetalleExclusividad": "No hay nada significativo."
    }
    out_ter = extractor_termino(seg_ter) if seg_ter else {
        "DuracionContrato": "NA", "FechaInicio": "", "FechaFin": "",
        "TerminacionUnilateral": "NO", "PenalidadTerminacion": "NO",
        "Preaviso": "NA", "DetalleTermino": "No hay nada significativo."
    }
    out_vig = extractor_vigencia(seg_vig) if seg_vig else {
        "TieneRenovacionAutomatica": "NO", "PeriodicidadRenovacion": "NA",
        "PreavisoNoRenovacion": "NA", "DetalleVigencia": "No hay nada significativo."
    }

    return {
        "DesembolsosPagos": out_des,
        "Exclusividad": out_exc,
        "TerminoContrato": out_ter,
        "VigenciaRenovacion": out_vig
    }

def main():
    # Verifica conexión con Ollama
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"❌ No puedo conectar con Ollama en {OLLAMA_BASE_URL}: {e}")
        return

    files = sorted(INPUT_DIR.glob("*.txt"))
    if not files:
        print(f"No encontré .txt en {INPUT_DIR.resolve()}")
        return

    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        result = process_contract_text(text)
        result["archivo"] = p.name
        out = OUTPUT_DIR / f"{p.stem}.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✓ {p.name} → {out}")

if __name__ == "__main__":
    main()