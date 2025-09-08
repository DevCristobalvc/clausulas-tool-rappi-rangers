#!/usr/bin/env python3
"""
Script optimizado para comparar cláusulas usando LLMs con structured outputs.
Versión rápida que procesa múltiples cláusulas en una sola llamada.
"""

import os
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Agregar el directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

# Cargar variables de entorno
load_dotenv()

# Configuración
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://rappi.litellm-prod.ai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5-mini")
MAX_CLAUSES_PER_BATCH = 10  # Procesar hasta 10 cláusulas por lote

# Rutas
BASE_DIR = Path(__file__).parent.parent
STANDARD_CLAUSES_FILE = BASE_DIR / "data" / "output" / "clausulas_estandar.json"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "output"
LOGS_DIR = BASE_DIR / "logs"

# Crear directorios si no existen
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'compare_clauses_fast.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cliente OpenAI
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# Modelos Pydantic para structured outputs
class ClauseMatch(BaseModel):
    """Modelo para representar una coincidencia de cláusula."""
    is_match: bool = Field(description="True si la cláusula coincide con alguna estándar")
    matched_clause_id: Optional[int] = Field(None, description="ID de la cláusula estándar que coincide")
    confidence: float = Field(description="Nivel de confianza de la coincidencia (0.0 a 1.0)")
    reason: str = Field(description="Explicación breve de por qué coincide o no coincide")
    similarity_type: str = Field(description="Tipo de similitud: 'exacta', 'semantica', 'parcial', 'no_coincide'")

class ClauseAnalysis(BaseModel):
    """Modelo para el análisis de una cláusula."""
    clause_id: int = Field(description="ID de la cláusula extraída")
    clause_title: str = Field(description="Título de la cláusula")
    match_result: ClauseMatch = Field(description="Resultado de la comparación")

class BatchAnalysis(BaseModel):
    """Modelo para el análisis de un lote de cláusulas."""
    analyses: List[ClauseAnalysis] = Field(description="Lista de análisis de cláusulas")

def load_standard_clauses() -> List[Dict]:
    """Carga las cláusulas estándar desde el archivo JSON."""
    try:
        with STANDARD_CLAUSES_FILE.open("r", encoding="utf-8") as f:
            clauses = json.load(f)
        logger.info(f"✅ Cargadas {len(clauses)} cláusulas estándar")
        return clauses
    except Exception as e:
        logger.error(f"❌ Error cargando cláusulas estándar: {e}")
        return []

def create_batch_prompt(standard_clauses: List[Dict], extracted_clauses: List[Dict]) -> str:
    """Crea el prompt para analizar un lote de cláusulas."""
    
    # Preparar cláusulas estándar (solo las primeras 15 para evitar límites de tokens)
    standard_sample = standard_clauses[:15]
    standard_text = "\n".join([
        f"ID {clause['id']}: {clause['clausula'][:150]}..."
        for clause in standard_sample
    ])
    
    # Preparar cláusulas extraídas
    extracted_text = "\n".join([
        f"ID {clause['id']}: {clause.get('title', 'Sin título')} - {clause.get('text', '')[:100]}..."
        for clause in extracted_clauses
    ])
    
    prompt = f"""
Eres un experto en análisis de contratos legales especializado en acuerdos comerciales de plataformas digitales.

TAREA: Analizar si cada cláusula extraída coincide con alguna cláusula estándar.

CLÁUSULAS ESTÁNDAR DISPONIBLES:
{standard_text}

CLÁUSULAS EXTRAÍDAS A ANALIZAR:
{extracted_text}

INSTRUCCIONES:
1. Para cada cláusula extraída, determina si coincide con alguna cláusula estándar
2. Una cláusula coincide si tiene el mismo significado legal y comercial
3. Considera variaciones en redacción, pero el contenido sustancial debe ser equivalente
4. Asigna un nivel de confianza (0.0 a 1.0) basado en qué tan clara es la coincidencia

CRITERIOS:
- EXACTA: Redacción muy similar, mismo contenido
- SEMANTICA: Significado equivalente, redacción diferente
- PARCIAL: Coincide parcialmente, pero falta información importante
- NO_COINCIDE: Contenido diferente o no relacionado

Responde con el análisis de cada cláusula extraída.
"""
    
    return prompt

def analyze_clauses_batch(standard_clauses: List[Dict], extracted_clauses: List[Dict]) -> List[ClauseAnalysis]:
    """Analiza un lote de cláusulas usando LLM."""
    
    logger.info(f"🤖 Analizando lote de {len(extracted_clauses)} cláusulas...")
    
    try:
        # Crear prompt
        prompt = create_batch_prompt(standard_clauses, extracted_clauses)
        
        # Llamar al modelo con structured output
        response = client.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de contratos legales especializado en acuerdos comerciales de plataformas digitales."},
                {"role": "user", "content": prompt}
            ],
            response_format=BatchAnalysis,
            temperature=0.1
        )
        
        # Extraer resultados
        parsed_response = response.choices[0].message.parsed
        analyses = parsed_response.analyses
        
        logger.info(f"✅ LLM analizó {len(analyses)} cláusulas en el lote")
        return analyses
        
    except Exception as e:
        logger.error(f"❌ Error en análisis con LLM: {e}")
        # Fallback: crear análisis básicos
        fallback_analyses = []
        for clause in extracted_clauses:
            fallback_analyses.append(ClauseAnalysis(
                clause_id=clause.get("id", 0),
                clause_title=clause.get("title", "Sin título"),
                match_result=ClauseMatch(
                    is_match=False,
                    matched_clause_id=None,
                    confidence=0.0,
                    reason=f"Error en procesamiento: {str(e)}",
                    similarity_type="no_coincide"
                )
            ))
        return fallback_analyses

def process_contract_file_fast(file_path: Path, standard_clauses: List[Dict]) -> Dict:
    """Procesa un archivo de contrato usando análisis por lotes."""
    
    logger.info(f"📄 Procesando: {file_path.name}")
    
    try:
        # Cargar datos del contrato
        with file_path.open("r", encoding="utf-8") as f:
            contract_data = json.load(f)
        
        extracted_clauses = contract_data.get("clausulas", [])
        if not extracted_clauses:
            logger.warning(f"⚠️ No se encontraron cláusulas en {file_path.name}")
            return {
                "archivo": file_path.name,
                "total_clausulas": 0,
                "clausulas_extra": [],
                "clausulas_encontradas": [],
                "estadisticas": {}
            }
        
        logger.info(f"  📋 Cláusulas extraídas: {len(extracted_clauses)}")
        
        # Procesar en lotes
        all_analyses = []
        
        for i in range(0, len(extracted_clauses), MAX_CLAUSES_PER_BATCH):
            batch = extracted_clauses[i:i + MAX_CLAUSES_PER_BATCH]
            logger.info(f"  🔄 Procesando lote {i//MAX_CLAUSES_PER_BATCH + 1}: cláusulas {i+1} a {min(i+MAX_CLAUSES_PER_BATCH, len(extracted_clauses))}")
            
            batch_analyses = analyze_clauses_batch(standard_clauses, batch)
            all_analyses.extend(batch_analyses)
        
        # Procesar resultados
        clausulas_extra = []
        clausulas_encontradas = []
        
        for analysis in all_analyses:
            # Encontrar la cláusula original
            original_clause = next(
                (c for c in extracted_clauses if c.get("id") == analysis.clause_id), 
                None
            )
            
            if not original_clause:
                continue
            
            clause_data = {
                "id": analysis.clause_id,
                "title": analysis.clause_title,
                "summary": original_clause.get("summary"),
                "text": original_clause.get("text", ""),
                "analysis": {
                    "is_match": analysis.match_result.is_match,
                    "matched_clause_id": analysis.match_result.matched_clause_id,
                    "confidence": analysis.match_result.confidence,
                    "reason": analysis.match_result.reason,
                    "similarity_type": analysis.match_result.similarity_type
                }
            }
            
            if analysis.match_result.is_match:
                clausulas_encontradas.append(clause_data)
            else:
                clausulas_extra.append(clause_data)
        
        # Calcular estadísticas
        total_clauses = len(extracted_clauses)
        found_count = len(clausulas_encontradas)
        extra_count = len(clausulas_extra)
        coverage_percentage = (found_count / total_clauses * 100) if total_clauses > 0 else 0
        
        # Calcular confianza promedio
        avg_confidence = sum(c["analysis"]["confidence"] for c in clausulas_encontradas) / len(clausulas_encontradas) if clausulas_encontradas else 0
        
        # Distribución por tipo de similitud
        similarity_dist = {}
        for clause in clausulas_encontradas:
            sim_type = clause["analysis"]["similarity_type"]
            similarity_dist[sim_type] = similarity_dist.get(sim_type, 0) + 1
        
        estadisticas = {
            "total_clausulas": total_clauses,
            "clausulas_encontradas": found_count,
            "clausulas_extra": extra_count,
            "porcentaje_cobertura": round(coverage_percentage, 2),
            "confianza_promedio": round(avg_confidence, 3),
            "distribucion_similitud": similarity_dist,
            "lotes_procesados": (len(extracted_clauses) + MAX_CLAUSES_PER_BATCH - 1) // MAX_CLAUSES_PER_BATCH
        }
        
        result = {
            "archivo": file_path.name,
            "total_clausulas": total_clauses,
            "clausulas_extra": clausulas_extra,
            "clausulas_encontradas": clausulas_encontradas,
            "estadisticas": estadisticas
        }
        
        logger.info(f"  ✅ Resultados: {found_count}/{total_clauses} coincidencias ({coverage_percentage:.1f}%)")
        logger.info(f"  📊 Confianza promedio: {avg_confidence:.3f}")
        logger.info(f"  🔄 Lotes procesados: {estadisticas['lotes_procesados']}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error procesando {file_path.name}: {e}")
        return {
            "archivo": file_path.name,
            "error": str(e),
            "total_clausulas": 0,
            "clausulas_extra": [],
            "clausulas_encontradas": [],
            "estadisticas": {}
        }

def main():
    """Función principal."""
    logger.info("🚀 INICIANDO COMPARACIÓN RÁPIDA CON LLM Y STRUCTURED OUTPUTS")
    logger.info(f"📦 Tamaño máximo de lote: {MAX_CLAUSES_PER_BATCH}")
    
    # Cargar cláusulas estándar
    standard_clauses = load_standard_clauses()
    if not standard_clauses:
        logger.error("❌ No se pudieron cargar las cláusulas estándar")
        return
    
    # Obtener archivos procesados (solo los que tienen cláusulas)
    processed_files = []
    for file_path in PROCESSED_DIR.glob("*.json"):
        # Filtrar archivos de diagnóstico
        if file_path.name not in [
            "diagnostico_completo_pdfs.json",
            "resumen_diagnostico_pdfs.json", 
            "diagnostico_pdfs_problematicos.json",
            "reporte_procesamiento.json"
        ]:
            processed_files.append(file_path)
    
    if not processed_files:
        logger.error("❌ No se encontraron archivos procesados")
        return
    
    logger.info(f"📁 Archivos a procesar: {len(processed_files)}")
    
    # Procesar cada archivo
    all_results = []
    total_stats = {
        "archivos_procesados": 0,
        "total_clausulas": 0,
        "total_encontradas": 0,
        "total_extra": 0,
        "cobertura_promedio": 0,
        "confianza_promedio": 0,
        "total_lotes": 0
    }
    
    for file_path in processed_files:
        result = process_contract_file_fast(file_path, standard_clauses)
        all_results.append(result)
        
        # Actualizar estadísticas totales
        if "error" not in result and "estadisticas" in result:
            total_stats["archivos_procesados"] += 1
            total_stats["total_clausulas"] += result["total_clausulas"]
            total_stats["total_encontradas"] += result["estadisticas"].get("clausulas_encontradas", 0)
            total_stats["total_extra"] += result["estadisticas"].get("clausulas_extra", 0)
            total_stats["total_lotes"] += result["estadisticas"].get("lotes_procesados", 0)
    
    # Calcular promedios
    if total_stats["archivos_procesados"] > 0:
        total_stats["cobertura_promedio"] = (
            total_stats["total_encontradas"] / total_stats["total_clausulas"] * 100
        ) if total_stats["total_clausulas"] > 0 else 0
        
        # Confianza promedio ponderada
        total_confidence = 0
        total_weight = 0
        for result in all_results:
            if "error" not in result and "estadisticas" in result and result.get("clausulas_encontradas"):
                weight = len(result["clausulas_encontradas"])
                confidence = result["estadisticas"].get("confianza_promedio", 0)
                total_confidence += confidence * weight
                total_weight += weight
        
        total_stats["confianza_promedio"] = total_confidence / total_weight if total_weight > 0 else 0
    
    # Guardar resultados
    output_file = OUTPUT_DIR / "comparacion_rapida_detallada.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # Guardar resumen
    summary_file = OUTPUT_DIR / "resumen_comparacion_rapida.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(total_stats, f, ensure_ascii=False, indent=2)
    
    # Crear lista de cláusulas extra con análisis detallado
    clausulas_extra_all = []
    for result in all_results:
        if "error" not in result:
            for clause in result["clausulas_extra"]:
                clausulas_extra_all.append({
                    "archivo": result["archivo"],
                    "id": clause["id"],
                    "title": clause["title"],
                    "summary": clause["summary"],
                    "text": clause["text"][:300] + "..." if len(clause["text"]) > 300 else clause["text"],
                    "analysis": clause["analysis"]
                })
    
    clausulas_extra_file = OUTPUT_DIR / "clausulas_extra_rapidas.json"
    with clausulas_extra_file.open("w", encoding="utf-8") as f:
        json.dump(clausulas_extra_all, f, ensure_ascii=False, indent=2)
    
    # Mostrar resumen
    logger.info("📊 RESUMEN DE COMPARACIÓN RÁPIDA:")
    logger.info(f"  📁 Archivos procesados: {total_stats['archivos_procesados']}")
    logger.info(f"  📋 Total cláusulas: {total_stats['total_clausulas']}")
    logger.info(f"  ✅ Cláusulas encontradas en base: {total_stats['total_encontradas']}")
    logger.info(f"  ⚠️ Cláusulas extra: {total_stats['total_extra']}")
    logger.info(f"  📊 Cobertura promedio: {total_stats['cobertura_promedio']:.1f}%")
    logger.info(f"  🎯 Confianza promedio: {total_stats['confianza_promedio']:.3f}")
    logger.info(f"  🔄 Total lotes procesados: {total_stats['total_lotes']}")
    logger.info(f"  📄 Resultados detallados: {output_file}")
    logger.info(f"  📊 Resumen: {summary_file}")
    logger.info(f"  📋 Cláusulas extra: {clausulas_extra_file}")
    
    logger.info("🎉 Proceso completado exitosamente!")

if __name__ == "__main__":
    main()
