#!/usr/bin/env python3
"""
Script para comparar las cláusulas extraídas de cada PDF con las cláusulas estándar
y identificar cuáles no están presentes en el archivo estándar.
"""

import json
import logging
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Dict, Set

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comparacion_clausulas.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def similarity(a: str, b: str) -> float:
    """Calcula la similitud entre dos textos."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def normalize_text(text: str) -> str:
    """Normaliza el texto para comparación."""
    # Remover espacios extra, saltos de línea y caracteres especiales
    import re
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

def find_best_match(target_clause: str, standard_clauses: List[Dict], threshold: float = 0.7) -> Dict:
    """
    Encuentra la mejor coincidencia para una cláusula en las cláusulas estándar.
    """
    target_normalized = normalize_text(target_clause)
    best_match = None
    best_score = 0
    
    for std_clause in standard_clauses:
        std_text = std_clause.get("clausula", "")
        std_normalized = normalize_text(std_text)
        
        # Calcular similitud
        score = similarity(target_normalized, std_normalized)
        
        if score > best_score:
            best_score = score
            best_match = {
                "id": std_clause.get("id"),
                "clausula_estandar": std_text,
                "score": score,
                "clausula_extraida": target_clause
            }
    
    if best_score >= threshold:
        return best_match
    else:
        return None

def compare_clauses_with_standard(pdf_file: Path, standard_clauses: List[Dict]) -> Dict:
    """
    Compara las cláusulas de un PDF con las cláusulas estándar.
    """
    logger.info(f"📄 Comparando cláusulas de {pdf_file.name}...")
    
    try:
        with pdf_file.open("r", encoding="utf-8") as f:
            pdf_data = json.load(f)
    except Exception as e:
        logger.error(f"Error leyendo {pdf_file.name}: {e}")
        return {}
    
    extracted_clauses = pdf_data.get("clausulas", [])
    if not extracted_clauses:
        logger.warning(f"No se encontraron cláusulas en {pdf_file.name}")
        return {}
    
    logger.info(f"  📋 Cláusulas extraídas: {len(extracted_clauses)}")
    
    # Comparar cada cláusula extraída
    matches = []
    no_matches = []
    used_standard_ids = set()
    
    for clause in extracted_clauses:
        clause_text = clause.get("text", "")
        if not clause_text:
            continue
            
        # Buscar mejor coincidencia
        match = find_best_match(clause_text, standard_clauses, threshold=0.6)
        
        if match:
            matches.append({
                "clausula_extraida": {
                    "id": clause.get("id"),
                    "text": clause_text,
                    "title": clause.get("title"),
                    "summary": clause.get("summary")
                },
                "clausula_estandar": {
                    "id": match["id"],
                    "text": match["clausula_estandar"]
                },
                "similitud": match["score"]
            })
            used_standard_ids.add(match["id"])
        else:
            no_matches.append({
                "id": clause.get("id"),
                "text": clause_text,
                "title": clause.get("title"),
                "summary": clause.get("summary")
            })
    
    # Identificar cláusulas estándar no encontradas
    all_standard_ids = {clause["id"] for clause in standard_clauses}
    missing_standard_ids = all_standard_ids - used_standard_ids
    missing_standard_clauses = [
        clause for clause in standard_clauses 
        if clause["id"] in missing_standard_ids
    ]
    
    result = {
        "archivo": pdf_file.name,
        "total_clausulas_extraidas": len(extracted_clauses),
        "clausulas_encontradas": len(matches),
        "clausulas_no_encontradas": len(no_matches),
        "clausulas_estandar_faltantes": len(missing_standard_clauses),
        "porcentaje_cobertura": (len(matches) / len(extracted_clauses) * 100) if extracted_clauses else 0,
        "matches": matches,
        "no_matches": no_matches,
        "missing_standard_clauses": missing_standard_clauses
    }
    
    logger.info(f"  ✅ Cláusulas encontradas: {len(matches)}")
    logger.info(f"  ❌ Cláusulas no encontradas: {len(no_matches)}")
    logger.info(f"  📊 Cobertura: {result['porcentaje_cobertura']:.1f}%")
    
    return result

def main():
    """Función principal para comparar todas las cláusulas."""
    logger.info("🔍 INICIANDO COMPARACIÓN DE CLÁUSULAS")
    
    # Cargar cláusulas estándar
    standard_file = Path("clausulas_estandar.json")
    if not standard_file.exists():
        logger.error("No se encontró el archivo clausulas_estandar.json")
        return
    
    try:
        with standard_file.open("r", encoding="utf-8") as f:
            standard_clauses = json.load(f)
        logger.info(f"📋 Cláusulas estándar cargadas: {len(standard_clauses)}")
    except Exception as e:
        logger.error(f"Error cargando cláusulas estándar: {e}")
        return
    
    # Buscar archivos JSON procesados
    output_dir = Path("salida")
    if not output_dir.exists():
        logger.error("No se encontró el directorio salida/")
        return
    
    # Filtrar archivos JSON que contengan cláusulas (excluir archivos de diagnóstico)
    json_files = [
        f for f in output_dir.glob("*.json") 
        if f.name not in ["diagnostico_completo_pdfs.json", "resumen_diagnostico_pdfs.json", 
                          "diagnostico_pdfs_problematicos.json", "reporte_procesamiento.json"]
    ]
    
    if not json_files:
        logger.error("No se encontraron archivos JSON procesados en salida/")
        return
    
    logger.info(f"📁 Archivos a comparar: {len(json_files)}")
    
    # Comparar cada archivo
    all_comparisons = []
    summary_stats = {
        "total_archivos": len(json_files),
        "total_clausulas_extraidas": 0,
        "total_clausulas_encontradas": 0,
        "total_clausulas_no_encontradas": 0,
        "archivos_con_clausulas": 0
    }
    
    for json_file in json_files:
        comparison = compare_clauses_with_standard(json_file, standard_clauses)
        if comparison:
            all_comparisons.append(comparison)
            summary_stats["total_clausulas_extraidas"] += comparison["total_clausulas_extraidas"]
            summary_stats["total_clausulas_encontradas"] += comparison["clausulas_encontradas"]
            summary_stats["total_clausulas_no_encontradas"] += comparison["clausulas_no_encontradas"]
            if comparison["total_clausulas_extraidas"] > 0:
                summary_stats["archivos_con_clausulas"] += 1
    
    # Guardar comparaciones detalladas
    comparison_file = output_dir / "comparacion_clausulas_detallada.json"
    with comparison_file.open("w", encoding="utf-8") as f:
        json.dump(all_comparisons, f, ensure_ascii=False, indent=2)
    
    # Crear resumen de cláusulas no encontradas
    no_matches_summary = []
    for comparison in all_comparisons:
        for no_match in comparison["no_matches"]:
            no_matches_summary.append({
                "archivo": comparison["archivo"],
                "id": no_match["id"],
                "title": no_match["title"],
                "summary": no_match["summary"],
                "text": no_match["text"][:200] + "..." if len(no_match["text"]) > 200 else no_match["text"]
            })
    
    no_matches_file = output_dir / "clausulas_no_encontradas.json"
    with no_matches_file.open("w", encoding="utf-8") as f:
        json.dump(no_matches_summary, f, ensure_ascii=False, indent=2)
    
    # Calcular estadísticas finales
    if summary_stats["total_clausulas_extraidas"] > 0:
        summary_stats["porcentaje_cobertura_general"] = (
            summary_stats["total_clausulas_encontradas"] / 
            summary_stats["total_clausulas_extraidas"] * 100
        )
    else:
        summary_stats["porcentaje_cobertura_general"] = 0
    
    # Guardar resumen
    summary_file = output_dir / "resumen_comparacion_clausulas.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)
    
    # Mostrar resumen
    logger.info("📊 RESUMEN DE COMPARACIÓN:")
    logger.info(f"  📁 Archivos procesados: {summary_stats['total_archivos']}")
    logger.info(f"  📋 Total cláusulas extraídas: {summary_stats['total_clausulas_extraidas']}")
    logger.info(f"  ✅ Cláusulas encontradas en estándar: {summary_stats['total_clausulas_encontradas']}")
    logger.info(f"  ❌ Cláusulas NO encontradas: {summary_stats['total_clausulas_no_encontradas']}")
    logger.info(f"  📊 Cobertura general: {summary_stats['porcentaje_cobertura_general']:.1f}%")
    logger.info(f"  📄 Comparación detallada: {comparison_file}")
    logger.info(f"  📋 Cláusulas no encontradas: {no_matches_file}")
    logger.info(f"  📊 Resumen: {summary_file}")
    
    # Mostrar cláusulas no encontradas por archivo
    logger.info("\n📋 CLAÚSULAS NO ENCONTRADAS POR ARCHIVO:")
    for comparison in all_comparisons:
        if comparison["no_matches"]:
            logger.info(f"  📄 {comparison['archivo']}: {len(comparison['no_matches'])} cláusulas")
            for no_match in comparison["no_matches"][:3]:  # Mostrar solo las primeras 3
                logger.info(f"    - {no_match.get('title', 'Sin título')}: {no_match['text'][:100]}...")
            if len(comparison["no_matches"]) > 3:
                logger.info(f"    ... y {len(comparison['no_matches']) - 3} más")

if __name__ == "__main__":
    main()
