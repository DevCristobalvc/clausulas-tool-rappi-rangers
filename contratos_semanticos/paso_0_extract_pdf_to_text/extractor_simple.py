# -*- coding: utf-8 -*-
"""
Extractor SIMPLE (sin embeddings) para contratos en outputs/*.txt
- Reglas básicas por palabras clave
- JSON por contrato en json_simple/
- Campos exactamente como los pediste
"""
import re, os, json, unicodedata
from pathlib import Path

# dateparser es opcional; si no está, las fechas quedan vacías
try:
    import dateparser
except Exception:
    dateparser = None

INPUT_DIR = Path("outputs")
OUTPUT_DIR = Path("json_simple")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- utilidades --------
def norm(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def has_any(t: str, kws):
    t = t.lower()
    return any(k.lower() in t for k in kws)

def negated(t: str, kw: str, window=30):
    t_low = t.lower(); kw = kw.lower()
    for m in re.finditer(re.escape(kw), t_low):
        start = max(0, m.start()-window)
        ctx = t_low[start:m.end()]
        if "no " in ctx or "sin " in ctx:
            return True
    return False

def parse_date(s: str):
    if not s or not dateparser: return None
    dt = dateparser.parse(s, languages=["es", "en"])
    return dt.strftime("%Y-%m-%d") if dt else None

def find_dates(text: str):
    # ISO directo
    iso = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if iso:
        fi = parse_date(iso[0]); ff = parse_date(iso[1]) if len(iso) > 1 else None
        return fi, ff
    # "12 de marzo de 2024"
    f = re.findall(r"\b(\d{1,2}\s+de\s+[A-Za-záéíóúñ]+(?:\s+de)?\s+\d{4})\b", text, flags=re.I)
    if f:
        fi = parse_date(f[0]); ff = parse_date(f[1]) if len(f) > 1 else None
        return fi, ff
    return None, None

def find_preaviso(text: str):
    t = text.lower()
    m = re.search(r"(?:preaviso|antelación|avisar con)\s*(\d{1,4})\s*(d[ií]a[s]?|mes[es]?)", t, flags=re.I)
    if not m:
        m = re.search(r"(\d{1,4})\s*(d[ií]a[s]?|mes[es]?)\s+de\s+preaviso", t, flags=re.I)
    if m:
        n, u = m.group(1), m.group(2)
        u = "días" if u.startswith("d") else "meses"
        return f"{n} {u}"
    return "NA"

def find_duracion(text: str):
    t = text.lower()
    m = re.search(r"(\d{1,3})\s*(mes|meses|año|años)", t)
    if m:
        num, u = m.group(1), m.group(2)
        u = "meses" if "mes" in u else "años"
        return f"{num} {u}"
    if "indefinid" in t:
        return "Indefinido"
    return "NA"

def excerpt(text: str, keywords, radius=350):
    t_low = text.lower()
    for kw in keywords:
        i = t_low.find(kw.lower())
        if i != -1:
            a = max(0, i - radius); b = min(len(text), i + len(kw) + radius)
            return norm(text[a:b])
    # fallback: primeras 400
    return norm(text[:400]) if text else "No hay nada significativo."

# -------- extractores por sección --------
def extract_desembolsos(text: str):
    t = text.lower()
    pos = ["bono de firma","upfront","bono de apertura","pago inicial",
           "desembolso extraordinario","compensación adicional","pago único",
           "incentivo de firma","pago por suscripción","pago de bienvenida"]
    neg = ["no habrá pagos","sin pagos","no se realizan pagos","solo órdenes","únicamente órdenes"]
    tiene = "SI" if any(p in t for p in pos) and not any(n in t for n in neg) else "NO"

    # Periodicidad
    per_map = {
        "mensual":"Mensual","quincenal":"Quincenal","semanal":"Semanal",
        "contra hitos":"Contra Hitos","por hitos":"Contra Hitos",
        "a la entrega":"Contra Hitos","único":"Único","una sola vez":"Único"
    }
    per = "NA"
    for k,v in per_map.items():
        if k in t: per = v; break

    # Forma de pago
    forma = "NA" if tiene=="NO" else "Otro"
    if "transferencia" in t: forma = "Transferencia"
    elif "efectivo" in t: forma = "Efectivo"
    elif "cheque" in t: forma = "Cheque"
    elif "pse" in t: forma = "PSE"

    condiciones = excerpt(text, ["pago","desembolso","bono","upfront"])
    detalle = condiciones if condiciones else "No hay nada significativo."

    return {
        "TieneDesembolsos": tiene,
        "PeriodicidadPagos": per if tiene=="SI" else "NA",
        "CondicionesPago": detalle if tiene=="SI" else "No hay nada significativo.",
        "FormaPago": forma,
        "DetalleDesembolsos": detalle if tiene=="SI" else "No hay nada significativo."
    }

def extract_exclusividad(text: str):
    t = text.lower()
    tiene = "SI" if ("exclusiv" in t or "acuerdo exclusivo" in t or "no competencia" in t or "otras plataformas" in t) and not ("sin exclusiv" in t or "no exclusiv" in t) else "NO"

    # Alcance
    alcance = "NA"
    if "territor" in t: alcance = "Territorial"
    elif "plataforma" in t or "otras plataformas" in t or "uber eats" in t or "didi food" in t: alcance = "Plataforma"
    elif "cliente" in t: alcance = "Clientes"
    elif "producto" in t or "servicio" in t or "portafolio" in t: alcance = "Producto/Servicio"

    ruptura = "SI" if has_any(t, ["romper","terminar","rescindir","resolver","incumplimiento","causales"]) else "NO"
    condiciones = excerpt(text, ["exclusiv","plataforma","territorio","cliente","no competencia"])
    detalle = condiciones if condiciones else "No hay nada significativo."
    return {
        "TieneExclusividad": tiene,
        "AlcanceExclusividad": alcance if tiene=="SI" else "NA",
        "CondicionesExclusividad": detalle if tiene=="SI" else "No hay nada significativo.",
        "RupturaExclusividad": ruptura if tiene=="SI" else "NO",
        "DetalleExclusividad": detalle if tiene=="SI" else "No hay nada significativo."
    }

def extract_termino(text: str):
    dur = find_duracion(text)
    fi, ff = find_dates(text)
    t = text.lower()
    termin_unilat = "SI" if has_any(t, [
        "terminación unilateral","terminar unilateralmente","dar por terminado",
        "rescindir el contrato","terminación anticipada"
    ]) else "NO"
    penal = "SI" if has_any(t, ["cláusula penal","penalidad","multa","sanción","penalidad por terminación"]) else "NO"
    preav = find_preaviso(text)
    detalle = excerpt(text, ["terminación","plazo","vigencia","preaviso","rescindir"])
    return {
        "DuracionContrato": dur,
        "FechaInicio": fi or "",
        "FechaFin": ff or "",
        "TerminacionUnilateral": termin_unilat,
        "PenalidadTerminacion": penal,
        "Preaviso": preav,
        "DetalleTermino": detalle if detalle else "No hay nada significativo."
    }

def extract_vigencia(text: str):
    t = text.lower()
    # renovación automática con cuidado de "no"/"sin"
    tiene_auto = "SI" if ("renovación automática" in t or "renovacion automatica" in t) and not negated(t, "renovación") else "NO"

    per = "NA"
    if "anual" in t or "cada 12 meses" in t or "cada año" in t: per = "Anual"
    elif "semestral" in t or "cada 6 meses" in t: per = "Semestral"
    elif "mensual" in t: per = "Mensual"
    elif "renov" in t: per = "Otro"

    pre_no_ren = find_preaviso(text) if tiene_auto=="SI" else "NA"
    detalle = excerpt(text, ["vigencia","renovación","prórroga","extensión","duración"])
    return {
        "TieneRenovacionAutomatica": tiene_auto,
        "PeriodicidadRenovacion": per if tiene_auto=="SI" else "NA",
        "PreavisoNoRenovacion": pre_no_ren,
        "DetalleVigencia": detalle if "renov" in t or "vigencia" in t else "No hay nada significativo."
    }

# -------- pipeline --------
def process_text(txt: str):
    txt = norm(txt)
    return {
        "DesembolsosPagos": extract_desembolsos(txt),
        "Exclusividad": extract_exclusividad(txt),
        "TerminoContrato": extract_termino(txt),
        "VigenciaRenovacion": extract_vigencia(txt)
    }

def main():
    files = sorted(INPUT_DIR.glob("*.txt"))
    if not files:
        print(f"No encontré .txt en {INPUT_DIR.resolve()}")
        return
    for p in files:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        data = process_text(text)
        data["archivo"] = p.name
        out = OUTPUT_DIR / f"{p.stem}.json"
        with out.open("w", encoding="utf-8") as wf:
            json.dump(data, wf, ensure_ascii=False, indent=2)
        print(f"✓ {p.name} → {out}")

if __name__ == "__main__":
    main()