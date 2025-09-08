#!/usr/bin/env python3
"""
Script mejorado para comparar cláusulas usando LLMs con structured outputs.
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
        logging.FileHandler(LOGS_DIR / 'compare_clauses_llm.log'),
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
    reason: str = Field(description="Explicación de por qué coincide o no coincide")
    similarity_type: Optional[str] = Field(None, description="Tipo de similitud: 'exacta', 'semantica', 'parcial'")

class ClauseComparison(BaseModel):
    """Modelo para el resultado de comparación de una cláusula."""
    clause_id: int = Field(description="ID de la cláusula extraída")
    clause_text: str = Field(description="Texto de la cláusula extraída")
    clause_title: Optional[str] = Field(None, description="Título de la cláusula")
    clause_summary: Optional[str] = Field(None, description="Resumen de la cláusula")
    match_result: ClauseMatch = Field(description="Resultado de la comparación")

class ComparisonBatch(BaseModel):
    """Modelo para un lote de comparaciones."""
    comparisons: List[ClauseComparison] = Field(description="Lista de comparaciones")

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

def create_comparison_prompt(standard_clauses: List[Dict], extracted_clauses: List[Dict]) -> str:
    """Crea el prompt para la comparación usando LLM."""
    
    # Preparar cláusulas estándar para el prompt
    standard_text = "\n".join([
        f"ID {clause['id']}: {clause['clausula']}"
        for clause in standard_clauses
    ])
    
    # Preparar cláusulas extraídas para el prompt
    extracted_text = "\n".join([
        f"ID {clause['id']}: {clause.get('title', 'Sin título')} - {clause.get('text', '')[:200]}..."
        for clause in extracted_clauses
    ])
    
    prompt = f"""
Eres un experto en análisis de contratos legales especializado en acuerdos comerciales de plataformas digitales.

TAREA: Comparar cláusulas extraídas de contratos con un conjunto de cláusulas estándar y determinar si coinciden.

CLÁUSULAS ESTÁNDAR:
{standard_text}

CLÁUSULAS EXTRAÍDAS A COMPARAR:
{extracted_text}

INSTRUCCIONES:
1. Para cada cláusula extraída, determina si coincide con alguna cláusula estándar
2. Una cláusula coincide si:
   - Tiene el mismo significado legal y comercial
   - Establece las mismas obligaciones o derechos
   - Cubre el mismo aspecto contractual
3. Considera variaciones en redacción, pero el contenido sustancial debe ser equivalente
4. Asigna un nivel de confianza basado en qué tan clara es la coincidencia

CRITERIOS DE EVALUACIÓN:
- EXACTA: Redacción muy similar, mismo contenido
- SEMANTICA: Significado equivalente, redacción diferente
- PARCIAL: Coincide parcialmente, pero falta información importante
- NO COINCIDE: Contenido diferente o no relacionado

Responde con el análisis de cada cláusula extraída.
"""
    
    return prompt

def compare_clauses_with_llm(standard_clauses: List[Dict], extracted_clauses: List[Dict]) -> List[ClauseComparison]:
    """Compara cláusulas usando LLM con structured outputs."""
    
    logger.info(f"🤖 Comparando {len(extracted_clauses)} cláusulas con LLM...")
    
    try:
        # Crear prompt
        prompt = create_comparison_prompt(standard_clauses, extracted_clauses)
        
        # Llamar al modelo con structured output
        response = client.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de contratos legales."},
                {"role": "user", "content": prompt}
            ],
            response_format=ComparisonBatch,
            temperature=0.1  # Baja temperatura para consistencia
        )
        
        # Extraer resultados
        parsed_response = response.choices[0].message.parsed
        comparisons = parsed_response.comparisons
        
        logger.info(f"✅ LLM procesó {len(comparisons)} comparaciones")
        return comparisons
        
    except Exception as e:
        logger.error(f"❌ Error en comparación con LLM: {e}")
        return []

def process_contract_file(file_path: Path, standard_clauses: List[Dict]) -> Dict:
    """Procesa un archivo de contrato individual."""
    
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
                "matches": [],
                "no_matches": [],
                "estadisticas": {}
            }
        
        logger.info(f"  📋 Cláusulas extraídas: {len(extracted_clauses)}")
        
        # Comparar con LLM
        comparisons = compare_clauses_with_llm(standard_clauses, extracted_clauses)
        
        # Procesar resultados
        matches = []
        no_matches = []
        
        for comparison in comparisons:
            clause_data = {
                "id": comparison.clause_id,
                "text": comparison.clause_text,
                "title": comparison.clause_title,
                "summary": comparison.clause_summary,
                "match_result": {
                    "is_match": comparison.match_result.is_match,
                    "matched_clause_id": comparison.match_result.matched_clause_id,
                    "confidence": comparison.match_result.confidence,
                    "reason": comparison.match_result.reason,
                    "similarity_type": comparison.match_result.similarity_type
                }
            }
            
            if comparison.match_result.is_match:
                matches.append(clause_data)
            else:
                no_matches.append(clause_data)
        
        # Calcular estadísticas
        total_clauses = len(extracted_clauses)
        matched_count = len(matches)
        no_match_count = len(no_matches)
        coverage_percentage = (matched_count / total_clauses * 100) if total_clauses > 0 else 0
        
        # Calcular confianza promedio
        avg_confidence = sum(m["match_result"]["confidence"] for m in matches) / len(matches) if matches else 0
        
        estadisticas = {
            "total_clausulas": total_clauses,
            "clausulas_encontradas": matched_count,
            "clausulas_no_encontradas": no_match_count,
            "porcentaje_cobertura": round(coverage_percentage, 2),
            "confianza_promedio": round(avg_confidence, 3),
            "distribucion_similitud": {}
        }
        
        # Distribución por tipo de similitud
        for match in matches:
            sim_type = match["match_result"]["similarity_type"] or "no_especificado"
            estadisticas["distribucion_similitud"][sim_type] = estadisticas["distribucion_similitud"].get(sim_type, 0) + 1
        
        result = {
            "archivo": file_path.name,
            "total_clausulas": total_clauses,
            "matches": matches,
            "no_matches": no_matches,
            "estadisticas": estadisticas
        }
        
        logger.info(f"  ✅ Resultados: {matched_count}/{total_clauses} coincidencias ({coverage_percentage:.1f}%)")
        logger.info(f"  📊 Confianza promedio: {avg_confidence:.3f}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error procesando {file_path.name}: {e}")
        return {
            "archivo": file_path.name,
            "error": str(e),
            "total_clausulas": 0,
            "matches": [],
            "no_matches": [],
            "estadisticas": {}
        }

def main():
    """Función principal."""
    logger.info("🚀 INICIANDO COMPARACIÓN CON LLM Y STRUCTURED OUTPUTS")
    
    # Cargar cláusulas estándar
    standard_clauses = load_standard_clauses()
    if not standard_clauses:
        logger.error("❌ No se pudieron cargar las cláusulas estándar")
        return
    
    # Obtener archivos procesados
    processed_files = list(PROCESSED_DIR.glob("*.json"))
    if not processed_files:
        logger.error("❌ No se encontraron archivos procesados")
        return
    
    logger.info(f"📁 Archivos a procesar: {len(processed_files)}")
    
    # Procesar cada archivo
    all_results = []
    total_stats = {
        "archivos_procesados": 0,
        "total_clausulas": 0,
        "total_matches": 0,
        "total_no_matches": 0,
        "cobertura_promedio": 0,
        "confianza_promedio": 0
    }
    
    for file_path in processed_files:
        result = process_contract_file(file_path, standard_clauses)
        all_results.append(result)
        
        # Actualizar estadísticas totales
        if "error" not in result:
            total_stats["archivos_procesados"] += 1
            total_stats["total_clausulas"] += result["total_clausulas"]
            total_stats["total_matches"] += result["estadisticas"]["clausulas_encontradas"]
            total_stats["total_no_matches"] += result["estadisticas"]["clausulas_no_encontradas"]
    
    # Calcular promedios
    if total_stats["archivos_procesados"] > 0:
        total_stats["cobertura_promedio"] = (
            total_stats["total_matches"] / total_stats["total_clausulas"] * 100
        ) if total_stats["total_clausulas"] > 0 else 0
        
        # Confianza promedio ponderada
        total_confidence = 0
        total_weight = 0
        for result in all_results:
            if "error" not in result and result["matches"]:
                weight = len(result["matches"])
                confidence = result["estadisticas"]["confianza_promedio"]
                total_confidence += confidence * weight
                total_weight += weight
        
        total_stats["confianza_promedio"] = total_confidence / total_weight if total_weight > 0 else 0
    
    # Guardar resultados
    output_file = OUTPUT_DIR / "comparacion_llm_detallada.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # Guardar resumen
    summary_file = OUTPUT_DIR / "resumen_comparacion_llm.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(total_stats, f, ensure_ascii=False, indent=2)
    
    # Crear lista de cláusulas no encontradas
    no_matches_all = []
    for result in all_results:
        if "error" not in result:
            for no_match in result["no_matches"]:
                no_matches_all.append({
                    "archivo": result["archivo"],
                    "id": no_match["id"],
                    "title": no_match["title"],
                    "summary": no_match["summary"],
                    "text": no_match["text"][:300] + "..." if len(no_match["text"]) > 300 else no_match["text"],
                    "reason": no_match["match_result"]["reason"]
                })
    
    no_matches_file = OUTPUT_DIR / "clausulas_no_encontradas_llm.json"
    with no_matches_file.open("w", encoding="utf-8") as f:
        json.dump(no_matches_all, f, ensure_ascii=False, indent=2)
    
    # Mostrar resumen
    logger.info("📊 RESUMEN DE COMPARACIÓN CON LLM:")
    logger.info(f"  📁 Archivos procesados: {total_stats['archivos_procesados']}")
    logger.info(f"  📋 Total cláusulas: {total_stats['total_clausulas']}")
    logger.info(f"  ✅ Cláusulas encontradas: {total_stats['total_matches']}")
    logger.info(f"  ❌ Cláusulas no encontradas: {total_stats['total_no_matches']}")
    logger.info(f"  📊 Cobertura promedio: {total_stats['cobertura_promedio']:.1f}%")
    logger.info(f"  🎯 Confianza promedio: {total_stats['confianza_promedio']:.3f}")
    logger.info(f"  📄 Resultados detallados: {output_file}")
    logger.info(f"  📊 Resumen: {summary_file}")
    logger.info(f"  📋 Cláusulas no encontradas: {no_matches_file}")

if __name__ == "__main__":
    main()
