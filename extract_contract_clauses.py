import os
import json
import logging
from pathlib import Path
import pdfplumber
from openai import OpenAI  # asumes el mismo cliente que usaste antes
from pydantic import BaseModel
from typing import List, Optional

from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('contract_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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

def validate_config():
    """Valida la configuración antes de iniciar el procesamiento."""
    errors = []
    
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY no está configurada")
    if not OPENAI_BASE_URL:
        errors.append("OPENAI_BASE_URL no está configurada")
    if not MODEL_NAME:
        errors.append("MODEL_NAME no está configurada")
    
    if not INPUT_DIR.exists():
        errors.append(f"Directorio de entrada {INPUT_DIR} no existe")
    
    if errors:
        logger.error("❌ Errores de configuración:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("✅ Configuración validada correctamente")
    logger.info(f"  - Modelo: {MODEL_NAME}")
    logger.info(f"  - Base URL: {OPENAI_BASE_URL}")
    logger.info(f"  - Directorio entrada: {INPUT_DIR}")
    logger.info(f"  - Directorio salida: {OUTPUT_DIR}")
    return True

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
    logger.debug(f"Extrayendo texto de {pdf_path.name}...")
    paragraphs = []
    total_pages = 0
    pages_with_text = 0
    
    try:
        # Intentar diferentes métodos de extracción
        extraction_methods = [
            ("pdfplumber_default", lambda pdf: [page.extract_text() for page in pdf.pages]),
            ("pdfplumber_with_layout", lambda pdf: [page.extract_text(layout=True) for page in pdf.pages]),
            ("pdfplumber_words", lambda pdf: [page.extract_text(use_text_flow=True) for page in pdf.pages])
        ]
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.debug(f"PDF tiene {total_pages} páginas")
            
            # Intentar diferentes métodos de extracción
            texts_per_page = []
            successful_method = None
            
            for method_name, extract_func in extraction_methods:
                try:
                    logger.debug(f"Intentando método: {method_name}")
                    texts_per_page = extract_func(pdf)
                    
                    # Verificar si obtuvimos texto significativo
                    total_chars = sum(len(text or "") for text in texts_per_page)
                    if total_chars > 100:  # Al menos 100 caracteres
                        successful_method = method_name
                        logger.debug(f"Método exitoso: {method_name} ({total_chars} caracteres)")
                        break
                    else:
                        logger.debug(f"Método {method_name}: solo {total_chars} caracteres")
                        
                except Exception as e:
                    logger.debug(f"Método {method_name} falló: {e}")
                    continue
            
            if not successful_method:
                logger.warning(f"No se pudo extraer texto significativo de {pdf_path.name}")
                # Intentar extraer información básica del PDF
                try:
                    pdf_info = pdf.metadata
                    logger.debug(f"Metadatos del PDF: {pdf_info}")
                except:
                    pass
                return []
            
            # Procesar el texto extraído
            for page_num, text in enumerate(texts_per_page, 1):
                if not text or not text.strip():
                    logger.debug(f"Página {page_num}: sin texto")
                    continue
                
                pages_with_text += 1
                lines = [ln.rstrip() for ln in text.split("\n")]
                logger.debug(f"Página {page_num}: {len(lines)} líneas extraídas ({len(text)} caracteres)")
                
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
        
        logger.debug(f"Extracción completada: {pages_with_text}/{total_pages} páginas con texto, {len(paragraphs)} párrafos")
        
        # Estadísticas de longitud de párrafos
        if paragraphs:
            avg_length = sum(len(p) for p in paragraphs) / len(paragraphs)
            max_length = max(len(p) for p in paragraphs)
            min_length = min(len(p) for p in paragraphs)
            logger.debug(f"Estadísticas párrafos: avg={avg_length:.1f}, min={min_length}, max={max_length}")
            
            # Verificar calidad del texto extraído
            total_chars = sum(len(p) for p in paragraphs)
            if total_chars < 500:  # Menos de 500 caracteres totales
                logger.warning(f"Texto extraído muy corto ({total_chars} caracteres) - posible PDF escaneado")
        else:
            logger.warning(f"No se extrajeron párrafos de {pdf_path.name}")
        
        return paragraphs
        
    except Exception as e:
        logger.error(f"Error extrayendo texto de {pdf_path.name}: {e}")
        # Intentar obtener información adicional sobre el error
        try:
            import os
            file_size = os.path.getsize(pdf_path)
            logger.debug(f"Tamaño del archivo: {file_size} bytes")
        except:
            pass
        return []

def chunkify(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n], i

def diagnose_pdf(pdf_path: Path) -> dict:
    """
    Diagnostica un PDF para entender por qué no se puede extraer texto.
    """
    diagnosis = {
        "archivo": pdf_path.name,
        "existe": pdf_path.exists(),
        "tamaño_bytes": 0,
        "es_pdf_valido": False,
        "total_paginas": 0,
        "metodos_exitosos": [],
        "texto_extraido": "",
        "errores": []
    }
    
    if not pdf_path.exists():
        diagnosis["errores"].append("Archivo no existe")
        return diagnosis
    
    try:
        import os
        diagnosis["tamaño_bytes"] = os.path.getsize(pdf_path)
    except Exception as e:
        diagnosis["errores"].append(f"Error obteniendo tamaño: {e}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            diagnosis["es_pdf_valido"] = True
            diagnosis["total_paginas"] = len(pdf.pages)
            
            # Intentar diferentes métodos
            methods = [
                ("default", lambda p: p.extract_text()),
                ("layout", lambda p: p.extract_text(layout=True)),
                ("text_flow", lambda p: p.extract_text(use_text_flow=True)),
                ("words", lambda p: " ".join([word.get("text", "") for word in p.extract_words()]))
            ]
            
            for method_name, extract_func in methods:
                try:
                    page_texts = []
                    for page in pdf.pages:
                        text = extract_func(page)
                        if text:
                            page_texts.append(text)
                    
                    total_text = " ".join(page_texts)
                    if len(total_text.strip()) > 50:
                        diagnosis["metodos_exitosos"].append(method_name)
                        if not diagnosis["texto_extraido"]:
                            diagnosis["texto_extraido"] = total_text[:500] + "..." if len(total_text) > 500 else total_text
                            
                except Exception as e:
                    diagnosis["errores"].append(f"Método {method_name}: {e}")
            
            # Intentar obtener metadatos
            try:
                metadata = pdf.metadata
                diagnosis["metadatos"] = metadata
            except Exception as e:
                diagnosis["errores"].append(f"Error metadatos: {e}")
                
    except Exception as e:
        diagnosis["errores"].append(f"Error abriendo PDF: {e}")
    
    return diagnosis

def validate_clause_quality(clauses: List[dict]) -> dict:
    """
    Valida la calidad de las cláusulas extraídas y proporciona estadísticas.
    """
    if not clauses:
        return {"total": 0, "valid": 0, "with_title": 0, "with_summary": 0, "issues": []}
    
    issues = []
    valid_count = 0
    with_title = 0
    with_summary = 0
    
    for clause in clauses:
        # Validar estructura básica
        if not isinstance(clause, dict):
            issues.append(f"Cláusula no es diccionario: {clause}")
            continue
            
        # Validar campos requeridos
        if "id" not in clause or "text" not in clause or "is_clause" not in clause:
            issues.append(f"Cláusula falta campos requeridos: {clause.get('id', 'unknown')}")
            continue
            
        # Validar tipos
        if not isinstance(clause.get("id"), int):
            issues.append(f"ID no es entero: {clause.get('id')}")
            continue
            
        if not isinstance(clause.get("text"), str) or not clause["text"].strip():
            issues.append(f"Texto vacío o inválido: ID {clause.get('id')}")
            continue
            
        if not isinstance(clause.get("is_clause"), bool):
            issues.append(f"is_clause no es booleano: ID {clause.get('id')}")
            continue
        
        valid_count += 1
        
        # Contar títulos y resúmenes
        if clause.get("title") and clause["title"].strip():
            with_title += 1
            
        if clause.get("summary") and clause["summary"].strip():
            with_summary += 1
    
    return {
        "total": len(clauses),
        "valid": valid_count,
        "with_title": with_title,
        "with_summary": with_summary,
        "issues": issues
    }

# -------------------------
# Conversación con el LLM
# -------------------------

#REVISAR: los tipos de pronts que se puede usar parar que el modelo entienda bien

SYSTEM_PROMPT = """
Eres un experto en análisis de contratos comerciales y legales especializado en acuerdos de plataformas digitales y comercio electrónico. Tu tarea es identificar y extraer cláusulas contractuales específicas de documentos legales.

CONTEXTO ESPECÍFICO:
Estás analizando contratos relacionados con Rappi y sus aliados comerciales, incluyendo acuerdos de cooperación, términos y condiciones, y convenios modificatorios.

INSTRUCCIONES ESPECÍFICAS:
1. Identifica cláusulas contractuales que contengan:
   - Obligaciones y responsabilidades de las partes (Rappi y Aliado Comercial)
   - Términos de pago, comisiones, tarifas y porcentajes de uso de plataforma
   - Plazos, vigencia, términos de terminación y renovación automática
   - Penalidades, compensaciones, indemnizaciones y multas
   - Confidencialidad, protección de datos personales y seguridad cibernética
   - Exclusividad, restricciones y prohibiciones
   - Propiedad intelectual, derechos de marca y licencias
   - Resolución de conflictos, jurisdicción y ley aplicable
   - Modificaciones, enmiendas y notificaciones
   - Cumplimiento legal, regulaciones y auditorías
   - Modalidades de entrega (Pick-Up, delivery)
   - Balanceo de demanda y herramientas de gestión
   - Condiciones especiales y políticas de la plataforma

2. EXCLUYE de cláusulas:
   - Encabezados de documento (títulos, fechas, nombres de partes)
   - Índices y tablas de contenido
   - Pies de página, numeración y referencias
   - Definiciones básicas sin contenido contractual sustancial
   - Textos puramente informativos sin obligaciones
   - Firmas, espacios en blanco y elementos decorativos
   - Información de contacto sin contenido contractual

3. Para cada cláusula identificada, proporciona:
   - Un título descriptivo y conciso (máximo 6 palabras) que capture la esencia
   - Un resumen que capture la obligación o derecho principal (máximo 20 palabras)

FORMATO DE RESPUESTA:
Devuelve ÚNICAMENTE un JSON válido con esta estructura:
[
  {
    "id": número_del_párrafo,
    "text": "texto_completo_del_párrafo",
    "is_clause": true/false,
    "title": "título_descriptivo_o_null",
    "summary": "resumen_contractual_o_null"
  }
]

IMPORTANTE: 
- Responde SOLO con el JSON válido
- No agregues explicaciones, comentarios ni texto adicional
- Mantén la numeración exacta de los párrafos
- Sé preciso en la identificación de contenido contractual vs. informativo
- Prioriza cláusulas con contenido sustancial sobre definiciones básicas
"""

def call_model_for_paragraphs(paragraphs: List[str], start_idx: int):
    """
    Llama al modelo para un batch de párrafos.
    Devuelve lista de diccionarios (según spec).
    """
    logger.info(f"Procesando batch de {len(paragraphs)} párrafos (índices {start_idx+1} a {start_idx+len(paragraphs)})")
    
    # Construir prompt del usuario: numerar cada párrafo con su id absoluto
    lines = []
    for i, p in enumerate(paragraphs):
        pid = start_idx + i + 1  # ids empiezan en 1
        # limitamos tamaño por seguridad (el modelo puede trabajar con textos largos,
        # pero si hay párrafos gigantes quizá convenga truncar o manejarlo).
        lines.append(f"{pid}) {p}")
    user_content = "\n\n".join(lines)
    
    logger.debug(f"Prompt construido con {len(lines)} párrafos, longitud total: {len(user_content)} caracteres")

    # Usar responses.parse para obtener respuesta estructurada directamente
    try:
        logger.debug("Llamando a responses.parse con esquema Pydantic...")
        resp = client.responses.parse(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            text_format=ClausesList  # le pedimos mapear a nuestro esquema
        )
        
        logger.debug(f"Respuesta recibida: {type(resp)}")
        
        # Extraer datos del output_parsed
        if hasattr(resp, "output_parsed") and resp.output_parsed:
            logger.debug("Respuesta tiene output_parsed, extrayendo datos...")
            
            # pydantic v2 usa model_dump, v1 usa dict()
            try:
                parsed = resp.output_parsed.model_dump()
                logger.debug(f"Datos extraídos con model_dump: {type(parsed)}")
            except Exception as e:
                logger.debug(f"model_dump falló: {e}, intentando dict()...")
                try:
                    parsed = resp.output_parsed.dict()
                    logger.debug(f"Datos extraídos con dict(): {type(parsed)}")
                except Exception as e2:
                    logger.error(f"Ambos métodos de extracción fallaron: {e2}")
                    parsed = None
            
            if parsed and isinstance(parsed, dict) and "clauses" in parsed:
                clauses_count = len(parsed["clauses"])
                logger.info(f"✅ responses.parse exitoso: {clauses_count} cláusulas extraídas")
                return parsed["clauses"]
            else:
                logger.error("output_parsed no contiene estructura esperada o está vacío")
                logger.debug(f"Estructura recibida: {parsed}")
        else:
            logger.error("Respuesta no tiene output_parsed")
            
    except Exception as e:
        logger.error(f"Error en responses.parse: {e}")

    # Solo si responses.parse falla completamente, usar fallback conservador
    logger.warning("⚠️ responses.parse falló, usando fallback conservador")
    fallback = []
    for i, p in enumerate(paragraphs):
        pid = start_idx + i + 1
        # Heurística simple: si el párrafo es muy corto o contiene solo números/puntuación, probablemente no es cláusula
        is_likely_clause = len(p.strip()) > 50 and any(c.isalpha() for c in p)
        fallback.append({
            "id": pid, 
            "text": p, 
            "is_clause": is_likely_clause, 
            "title": None, 
            "summary": None
        })
    
    logger.info(f"Fallback conservador aplicado: {sum(1 for f in fallback if f['is_clause'])}/{len(fallback)} marcados como cláusulas")
    return fallback

# -------------------------
# Flujo principal
# -------------------------
def process_all_contracts():
    logger.info("=== INICIANDO PROCESAMIENTO DE CONTRATOS ===")
    
    pdf_files = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() == ".pdf"])
    if not pdf_files:
        logger.error(f"No se encontraron PDFs en {INPUT_DIR}")
        return

    logger.info(f"Encontrados {len(pdf_files)} archivos PDF para procesar")
    
    # Diagnosticar PDFs problemáticos
    logger.info("🔍 Diagnosticando PDFs...")
    problematic_pdfs = []
    for pdf_path in pdf_files:
        diagnosis = diagnose_pdf(pdf_path)
        if not diagnosis["metodos_exitosos"]:
            problematic_pdfs.append(diagnosis)
            logger.warning(f"  ⚠️ PDF problemático: {pdf_path.name}")
            logger.warning(f"    - Tamaño: {diagnosis['tamaño_bytes']} bytes")
            logger.warning(f"    - Páginas: {diagnosis['total_paginas']}")
            logger.warning(f"    - Errores: {len(diagnosis['errores'])}")
            if diagnosis['errores']:
                for error in diagnosis['errores'][:2]:  # Mostrar solo los primeros 2 errores
                    logger.warning(f"      * {error}")
    
    if problematic_pdfs:
        logger.warning(f"📋 {len(problematic_pdfs)} PDFs problemáticos detectados")
        # Guardar diagnóstico
        diagnosis_file = OUTPUT_DIR / "diagnostico_pdfs_problematicos.json"
        with diagnosis_file.open("w", encoding="utf-8") as f:
            json.dump(problematic_pdfs, f, ensure_ascii=False, indent=2)
        logger.info(f"📄 Diagnóstico guardado en: {diagnosis_file}")
    else:
        logger.info("✅ Todos los PDFs parecen válidos")
    
    total_processed = 0
    total_clauses = 0
    
    for pdf_path in pdf_files:
        logger.info(f"📄 Procesando {pdf_path.name}...")
        
        try:
            paragraphs = extract_paragraphs_from_pdf(pdf_path)
            logger.info(f"  📊 Párrafos detectados: {len(paragraphs)}")
            
            if not paragraphs:
                logger.warning(f"  ⚠️ No se extrajo texto del PDF {pdf_path.name}")
                continue

            # Batching: enviar por lotes de N párrafos para evitar tokens excesivos
            batch_size = 30
            all_clauses = []
            batch_count = 0
            
            for batch, start_idx in chunkify(paragraphs, batch_size):
                batch_count += 1
                logger.info(f"  🔄 Procesando batch {batch_count} (párrafos {start_idx+1} a {start_idx+len(batch)})...")
                
                result_items = call_model_for_paragraphs(batch, start_idx)
                if result_items:
                    logger.debug(f"    ✅ Batch procesado: {len(result_items)} elementos recibidos")
                    
                    # normalizar cada item esperado
                    valid_items = 0
                    for it in result_items:
                        # asegurar estructura mínima
                        try:
                            cid = int(it.get("id"))
                        except (ValueError, TypeError):
                            logger.warning(f"    ⚠️ ID inválido en item: {it.get('id')}")
                            cid = None
                        
                        text = it.get("text") or ""
                        is_clause = bool(it.get("is_clause"))
                        title = it.get("title") if "title" in it else None
                        summary = it.get("summary") if "summary" in it else None
                        
                        if cid and text:  # Solo agregar items válidos
                            all_clauses.append({
                                "id": cid,
                                "text": text,
                                "is_clause": is_clause,
                                "title": title,
                                "summary": summary
                            })
                            valid_items += 1
                    
                    logger.debug(f"    📝 Items válidos agregados: {valid_items}/{len(result_items)}")
                else:
                    logger.warning(f"    ⚠️ LLM devolvió vacío para batch {batch_count}, usando fallback heurístico")
                    for i, p in enumerate(batch):
                        all_clauses.append({
                            "id": start_idx + i + 1,
                            "text": p,
                            "is_clause": True,
                            "title": None,
                            "summary": None
                        })

            # Filtrar solo cláusulas detectadas
            detected_clauses = [c for c in all_clauses if c.get("is_clause")]
            logger.info(f"  📋 Cláusulas detectadas: {len(detected_clauses)}/{len(all_clauses)}")

            # Validar calidad de las cláusulas extraídas
            quality_stats = validate_clause_quality(detected_clauses)
            logger.info(f"  📈 Calidad: {quality_stats['valid']}/{quality_stats['total']} válidas")
            logger.info(f"  📝 Detalles: {quality_stats['with_title']} con título, {quality_stats['with_summary']} con resumen")
            
            if quality_stats['issues']:
                logger.warning(f"  ⚠️ Problemas de calidad encontrados: {len(quality_stats['issues'])}")
                for issue in quality_stats['issues'][:5]:  # Mostrar solo los primeros 5
                    logger.debug(f"    - {issue}")
                if len(quality_stats['issues']) > 5:
                    logger.debug(f"    - ... y {len(quality_stats['issues']) - 5} más")

            output_data = {
                "archivo": pdf_path.name,
                "total_parrafos_detectados": len(paragraphs),
                "total_clausulas_extraidas": len(detected_clauses),
                "total_items_procesados": len(all_clauses),
                "clausulas_con_titulo": quality_stats['with_title'],
                "clausulas_con_resumen": quality_stats['with_summary'],
                "clausulas_validas": quality_stats['valid'],
                "problemas_calidad": len(quality_stats['issues']),
                "clausulas": detected_clauses,
                # opcional: puedes guardar también todos los items para revisión
                "raw_items": all_clauses,
                "estadisticas_calidad": quality_stats
            }

            out_file = OUTPUT_DIR / f"{pdf_path.stem}.json"
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            logger.info(f"  ✅ Guardado: {out_file}")
            logger.info(f"  📊 Resumen: {len(detected_clauses)} cláusulas de {len(paragraphs)} párrafos\n")
            
            total_processed += 1
            total_clauses += len(detected_clauses)
            
        except Exception as e:
            logger.error(f"  ❌ Error procesando {pdf_path.name}: {e}")
            continue
    
    logger.info(f"=== PROCESAMIENTO COMPLETADO ===")
    logger.info(f"📊 Total: {total_processed} archivos procesados, {total_clauses} cláusulas extraídas")
    
    # Generar reporte de resumen
    generate_summary_report(total_processed, total_clauses)

def generate_summary_report(total_processed: int, total_clauses: int):
    """
    Genera un reporte de resumen del procesamiento.
    """
    logger.info("📋 Generando reporte de resumen...")
    
    # Leer todos los archivos JSON generados para estadísticas detalladas
    json_files = list(OUTPUT_DIR.glob("*.json"))
    if not json_files:
        logger.warning("No se encontraron archivos JSON para generar reporte")
        return
    
    total_files = len(json_files)
    total_paragraphs = 0
    total_items = 0
    total_valid_clauses = 0
    total_with_titles = 0
    total_with_summaries = 0
    total_quality_issues = 0
    
    for json_file in json_files:
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            total_paragraphs += data.get("total_parrafos_detectados", 0)
            total_items += data.get("total_items_procesados", 0)
            total_valid_clauses += data.get("clausulas_validas", 0)
            total_with_titles += data.get("clausulas_con_titulo", 0)
            total_with_summaries += data.get("clausulas_con_resumen", 0)
            total_quality_issues += data.get("problemas_calidad", 0)
            
        except Exception as e:
            logger.warning(f"Error leyendo {json_file.name}: {e}")
    
    # Calcular estadísticas
    avg_clauses_per_file = total_clauses / total_files if total_files > 0 else 0
    avg_paragraphs_per_file = total_paragraphs / total_files if total_files > 0 else 0
    title_coverage = (total_with_titles / total_clauses * 100) if total_clauses > 0 else 0
    summary_coverage = (total_with_summaries / total_clauses * 100) if total_clauses > 0 else 0
    
    # Generar reporte
    report = {
        "resumen_procesamiento": {
            "archivos_procesados": total_processed,
            "archivos_json_generados": total_files,
            "total_parrafos_extraidos": total_paragraphs,
            "total_items_procesados": total_items,
            "total_clausulas_extraidas": total_clauses,
            "total_clausulas_validas": total_valid_clauses,
            "promedio_clausulas_por_archivo": round(avg_clauses_per_file, 2),
            "promedio_parrafos_por_archivo": round(avg_paragraphs_per_file, 2)
        },
        "calidad_extraccion": {
            "clausulas_con_titulo": total_with_titles,
            "clausulas_con_resumen": total_with_summaries,
            "cobertura_titulos_porcentaje": round(title_coverage, 2),
            "cobertura_resumenes_porcentaje": round(summary_coverage, 2),
            "problemas_calidad_totales": total_quality_issues
        },
        "archivos_procesados": [f.name for f in json_files]
    }
    
    # Guardar reporte
    report_file = OUTPUT_DIR / "reporte_procesamiento.json"
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Log del resumen
    logger.info("📊 REPORTE DE RESUMEN:")
    logger.info(f"  📁 Archivos procesados: {total_processed}")
    logger.info(f"  📄 Total párrafos: {total_paragraphs}")
    logger.info(f"  📋 Total cláusulas: {total_clauses}")
    logger.info(f"  ✅ Cláusulas válidas: {total_valid_clauses}")
    logger.info(f"  📝 Con título: {total_with_titles} ({title_coverage:.1f}%)")
    logger.info(f"  📄 Con resumen: {total_with_summaries} ({summary_coverage:.1f}%)")
    logger.info(f"  ⚠️ Problemas calidad: {total_quality_issues}")
    logger.info(f"  📊 Promedio cláusulas/archivo: {avg_clauses_per_file:.1f}")
    logger.info(f"  📋 Reporte guardado en: {report_file}")


if __name__ == "__main__":
    if validate_config():
        process_all_contracts()
    else:
        logger.error("No se puede continuar debido a errores de configuración")
        exit(1)
