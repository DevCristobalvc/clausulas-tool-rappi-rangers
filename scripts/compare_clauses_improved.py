#!/usr/bin/env python3
"""
Script mejorado basado en identify_clausulas_base.py pero con structured outputs.
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
        logging.FileHandler(LOGS_DIR / 'compare_clauses_improved.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cliente OpenAI
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# Modelos Pydantic para structured outputs
class ClauseAnalysis(BaseModel):
    """Modelo para el análisis de una cláusula."""
    is_in_base: bool = Field(description="True si la cláusula corresponde a alguna cláusula base")
    matched_clause_id: Optional[int] = Field(None, description="ID de la cláusula base que coincide")
    confidence: float = Field(description="Nivel de confianza de la coincidencia (0.0 a 1.0)")
    reason: str = Field(description="Explicación detallada de por qué coincide o no coincide")
    similarity_type: str = Field(description="Tipo de similitud: 'exacta', 'semantica', 'parcial', 'no_coincide'")
    key_differences: List[str] = Field(default_factory=list, description="Principales diferencias encontradas")

class ClauseComparisonResult(BaseModel):
    """Modelo para el resultado de comparación."""
    clause_analysis: ClauseAnalysis = Field(description="Análisis de la cláusula")

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

def is_clause_in_base_improved(clause_text: str, standard_clauses: List[Dict]) -> ClauseAnalysis:
    """
    Versión mejorada de is_clause_in_base usando structured outputs.
    """
    logger.debug(f"   🤔 Evaluando cláusula: {clause_text[:80]}...")
    
    # Preparar cláusulas estándar para el prompt
    standard_text = "\n".join([
        f"ID {clause['id']}: {clause['clausula']}"
        for clause in standard_clauses
    ])
    
    prompt = f"""
Eres un experto en análisis de contratos legales especializado en acuerdos comerciales de plataformas digitales.

TAREA: Determinar si la cláusula proporcionada corresponde (aunque esté redactada diferente) a alguna de las cláusulas base.

CLÁUSULAS BASE DISPONIBLES:
{standard_text}

CLÁUSULA A EVALUAR:
"{clause_text}"

INSTRUCCIONES:
1. Analiza si la cláusula corresponde a alguna cláusula base
2. Una cláusula corresponde si:
   - Tiene el mismo significado legal y comercial
   - Establece las mismas obligaciones o derechos
   - Cubre el mismo aspecto contractual
   - Aplica a la misma relación comercial

3. Considera variaciones en redacción, pero el contenido sustancial debe ser equivalente
4. Asigna un nivel de confianza basado en qué tan clara es la coincidencia

CRITERIOS DE EVALUACIÓN:
- EXACTA: Redacción muy similar, mismo contenido legal
- SEMANTICA: Significado equivalente, redacción diferente pero misma obligación/derecho
- PARCIAL: Coincide parcialmente, pero falta información importante
- NO_COINCIDE: Contenido diferente o no relacionado

ELEMENTOS CLAVE A CONSIDERAR:
- Obligaciones de las partes (Rappi vs Aliado)
- Derechos y responsabilidades
- Términos de pago y comisiones
- Plazos y vigencia
- Penalidades y compensaciones
- Confidencialidad y protección de datos
- Exclusividad y restricciones
- Propiedad intelectual y licencias
- Resolución de conflictos y jurisdicción

Proporciona un análisis detallado de la cláusula.
"""
    
    try:
        # Llamar al modelo con structured output
        response = client.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de contratos legales especializado en acuerdos comerciales de plataformas digitales."},
                {"role": "user", "content": prompt}
            ],
            response_format=ClauseComparisonResult,
            temperature=0.1
        )
        
        # Extraer resultado
        parsed_response = response.choices[0].message.parsed
        analysis = parsed_response.clause_analysis
        
        logger.debug(f"   ➡️ Resultado: {'COINCIDE' if analysis.is_in_base else 'NO COINCIDE'} (confianza: {analysis.confidence:.2f})")
        
        return analysis
        
    except Exception as e:
        logger.error(f"   ❌ Error en análisis: {e}")
        # Fallback: análisis básico
        return ClauseAnalysis(
            is_in_base=False,
            matched_clause_id=None,
            confidence=0.0,
            reason=f"Error en procesamiento: {str(e)}",
            similarity_type="no_coincide",
            key_differences=[f"Error técnico: {str(e)}"]
        )

def process_contract_file(file_path: Path, standard_clauses: List[Dict]) -> Dict:
    """Procesa un archivo de contrato individual."""
    
    logger.info(f"📂 Procesando archivo: {file_path.name}")
    
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
        
        logger.info(f"   📑 Total cláusulas detectadas: {len(extracted_clauses)}")
        
        clausulas_extra = []
        clausulas_encontradas = []
        
        for idx, clause in enumerate(extracted_clauses, start=1):
            text = clause.get("text", "").strip()
            title = clause.get("title", "Sin título")
            
            logger.info(f"   🔍 Cláusula {idx}/{len(extracted_clauses)} → {title}")
            
            # Analizar con LLM
            analysis = is_clause_in_base_improved(text, standard_clauses)
            
            clause_data = {
                "id": clause.get("id", idx),
                "title": title,
                "summary": clause.get("summary"),
                "text": text,
                "analysis": {
                    "is_in_base": analysis.is_in_base,
                    "matched_clause_id": analysis.matched_clause_id,
                    "confidence": analysis.confidence,
                    "reason": analysis.reason,
                    "similarity_type": analysis.similarity_type,
                    "key_differences": analysis.key_differences
                }
            }
            
            if analysis.is_in_base:
                clausulas_encontradas.append(clause_data)
                logger.info(f"   ✅ Coincide con cláusula base ID {analysis.matched_clause_id} (confianza: {analysis.confidence:.2f})")
            else:
                clausulas_extra.append(clause_data)
                logger.info(f"   ⚠️ Marcada como EXTRA - {analysis.similarity_type}")
            
            logger.info(f"   📝 Razón: {analysis.reason[:100]}...")
        
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
            "distribucion_similitud": similarity_dist
        }
        
        result = {
            "archivo": file_path.name,
            "total_clausulas": total_clauses,
            "clausulas_extra": clausulas_extra,
            "clausulas_encontradas": clausulas_encontradas,
            "estadisticas": estadisticas
        }
        
        logger.info(f"📌 Archivo procesado: {file_path.name}")
        logger.info(f"✅ Encontradas {found_count} cláusulas base, {extra_count} cláusulas extra")
        logger.info(f"📊 Cobertura: {coverage_percentage:.1f}%, Confianza promedio: {avg_confidence:.3f}")
        
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
    logger.info("🚀 INICIANDO IDENTIFICACIÓN DE CLAÚSULAS CON LLM MEJORADO")
    
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
        "total_encontradas": 0,
        "total_extra": 0,
        "cobertura_promedio": 0,
        "confianza_promedio": 0
    }
    
    for file_path in processed_files:
        result = process_contract_file(file_path, standard_clauses)
        all_results.append(result)
        
        # Actualizar estadísticas totales
        if "error" not in result and "estadisticas" in result:
            total_stats["archivos_procesados"] += 1
            total_stats["total_clausulas"] += result["total_clausulas"]
            total_stats["total_encontradas"] += result["estadisticas"].get("clausulas_encontradas", 0)
            total_stats["total_extra"] += result["estadisticas"].get("clausulas_extra", 0)
    
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
    output_file = OUTPUT_DIR / "identificacion_clausulas_mejorada.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # Guardar resumen
    summary_file = OUTPUT_DIR / "resumen_identificacion_mejorada.json"
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
    
    clausulas_extra_file = OUTPUT_DIR / "clausulas_extra_detalladas.json"
    with clausulas_extra_file.open("w", encoding="utf-8") as f:
        json.dump(clausulas_extra_all, f, ensure_ascii=False, indent=2)
    
    # Mostrar resumen
    logger.info("📊 RESUMEN DE IDENTIFICACIÓN MEJORADA:")
    logger.info(f"  📁 Archivos procesados: {total_stats['archivos_procesados']}")
    logger.info(f"  📋 Total cláusulas: {total_stats['total_clausulas']}")
    logger.info(f"  ✅ Cláusulas encontradas en base: {total_stats['total_encontradas']}")
    logger.info(f"  ⚠️ Cláusulas extra: {total_stats['total_extra']}")
    logger.info(f"  📊 Cobertura promedio: {total_stats['cobertura_promedio']:.1f}%")
    logger.info(f"  🎯 Confianza promedio: {total_stats['confianza_promedio']:.3f}")
    logger.info(f"  📄 Resultados detallados: {output_file}")
    logger.info(f"  📊 Resumen: {summary_file}")
    logger.info(f"  📋 Cláusulas extra detalladas: {clausulas_extra_file}")
    
    logger.info("🎉 Proceso completado. Revisa los archivos de salida para los resultados.")

if __name__ == "__main__":
    main()
