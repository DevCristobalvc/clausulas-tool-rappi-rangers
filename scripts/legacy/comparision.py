#!/usr/bin/env python3
"""
Script para comparar cláusulas extraídas con cláusulas estándar
"""

import json
import logging
from pathlib import Path
from difflib import SequenceMatcher
import re

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

def normalize_text(text):
    """Normaliza el texto para comparación"""
    if not text:
        return ""
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Remover caracteres especiales y espacios extra
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remover palabras comunes que no aportan significado
    stop_words = ['el', 'la', 'los', 'las', 'de', 'del', 'en', 'con', 'por', 'para', 'que', 'se', 'a', 'al', 'un', 'una']
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    
    return ' '.join(words)

def similarity_score(text1, text2):
    """Calcula similitud entre dos textos"""
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    if not norm1 or not norm2:
        return 0.0
    
    return SequenceMatcher(None, norm1, norm2).ratio()

def find_matching_clause(extracted_text, standard_clauses, threshold=0.7):
    """Encuentra la cláusula estándar más similar"""
    best_match = None
    best_score = 0.0
    
    for std_clause in standard_clauses:
        score = similarity_score(extracted_text, std_clause['clausula'])
        if score > best_score and score >= threshold:
            best_score = score
            best_match = {
                'id': std_clause['id'],
                'clausula': std_clause['clausula'],
                'score': score
            }
    
    return best_match

def compare_clauses_with_standard():
    """Compara todas las cláusulas extraídas con las estándar"""
    
    # Cargar cláusulas estándar
    logger.info("�� Cargando cláusulas estándar...")
    try:
        with open('clausulas_estandar.json', 'r', encoding='utf-8') as f:
            standard_clauses = json.load(f)
        logger.info(f"✅ Cargadas {len(standard_clauses)} cláusulas estándar")
    except Exception as e:
        logger.error(f"❌ Error cargando cláusulas estándar: {e}")
        return
    
    # Directorio de salida
    output_dir = Path("salida")
    if not output_dir.exists():
        logger.error("❌ Directorio 'salida' no existe")
        return
    
    # Buscar archivos JSON procesados (excluir archivos de diagnóstico)
    json_files = [f for f in output_dir.glob("*.json") 
                  if not f.name.startswith("diagnostico") 
                  and not f.name.startswith("reporte")
                  and not f.name.startswith("resumen")]
    
    logger.info(f"📁 Encontrados {len(json_files)} archivos procesados")
    
    # Resultados de comparación
    comparison_results = {}
    total_extracted = 0
    total_matched = 0
    total_unmatched = 0
    
    for json_file in json_files:
        logger.info(f"🔍 Analizando {json_file.name}...")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_name = data.get('archivo', json_file.stem)
            extracted_clauses = data.get('clausulas', [])
            
            logger.info(f"  📄 {file_name}: {len(extracted_clauses)} cláusulas extraídas")
            
            # Comparar cada cláusula extraída
            file_results = {
                'archivo': file_name,
                'total_clausulas': len(extracted_clauses),
                'clausulas_matched': [],
                'clausulas_unmatched': [],
                'estadisticas': {
                    'matched': 0,
                    'unmatched': 0,
                    'porcentaje_match': 0.0
                }
            }
            
            for clause in extracted_clauses:
                clause_text = clause.get('text', '')
                clause_title = clause.get('title', 'Sin título')
                
                # Buscar coincidencia en cláusulas estándar
                match = find_matching_clause(clause_text, standard_clauses)
                
                if match:
                    file_results['clausulas_matched'].append({
                        'id': clause.get('id'),
                        'title': clause_title,
                        'text': clause_text[:200] + "..." if len(clause_text) > 200 else clause_text,
                        'matched_with': {
                            'id': match['id'],
                            'score': round(match['score'], 3),
                            'standard_text': match['clausula'][:200] + "..." if len(match['clausula']) > 200 else match['clausula']
                        }
                    })
                    file_results['estadisticas']['matched'] += 1
                else:
                    file_results['clausulas_unmatched'].append({
                        'id': clause.get('id'),
                        'title': clause_title,
                        'text': clause_text[:200] + "..." if len(clause_text) > 200 else clause_text,
                        'reason': 'No se encontró coincidencia con cláusulas estándar'
                    })
                    file_results['estadisticas']['unmatched'] += 1
            
            # Calcular estadísticas
            total_clauses = file_results['estadisticas']['matched'] + file_results['estadisticas']['unmatched']
            if total_clauses > 0:
                file_results['estadisticas']['porcentaje_match'] = round(
                    (file_results['estadisticas']['matched'] / total_clauses) * 100, 2
                )
            
            comparison_results[file_name] = file_results
            
            # Actualizar totales
            total_extracted += len(extracted_clauses)
            total_matched += file_results['estadisticas']['matched']
            total_unmatched += file_results['estadisticas']['unmatched']
            
            logger.info(f"  ✅ {file_results['estadisticas']['matched']} coincidencias, {file_results['estadisticas']['unmatched']} sin coincidencia")
            
        except Exception as e:
            logger.error(f"❌ Error procesando {json_file.name}: {e}")
            continue
    
    # Generar reporte final
    logger.info("📊 Generando reporte de comparación...")
    
    # Crear resumen
    summary = {
        'fecha_comparacion': str(Path().cwd()),
        'archivos_procesados': len(json_files),
        'total_clausulas_extraidas': total_extracted,
        'total_clausulas_matched': total_matched,
        'total_clausulas_unmatched': total_unmatched,
        'porcentaje_match_global': round((total_matched / total_extracted * 100), 2) if total_extracted > 0 else 0,
        'archivos': comparison_results
    }
    
    # Guardar resultados detallados
    output_file = output_dir / "comparacion_clausulas_detallada.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Guardar solo cláusulas no encontradas
    unmatched_only = {}
    for file_name, results in comparison_results.items():
        if results['clausulas_unmatched']:
            unmatched_only[file_name] = {
                'archivo': file_name,
                'clausulas_no_encontradas': results['clausulas_unmatched']
            }
    
    unmatched_file = output_dir / "clausulas_no_encontradas.json"
    with open(unmatched_file, 'w', encoding='utf-8') as f:
        json.dump(unmatched_only, f, ensure_ascii=False, indent=2)
    
    # Imprimir resumen
    logger.info("=" * 60)
    logger.info("📋 RESUMEN DE COMPARACIÓN")
    logger.info("=" * 60)
    logger.info(f"📁 Archivos procesados: {len(json_files)}")
    logger.info(f"📄 Total cláusulas extraídas: {total_extracted}")
    logger.info(f"✅ Cláusulas con coincidencia: {total_matched}")
    logger.info(f"❌ Cláusulas sin coincidencia: {total_unmatched}")
    logger.info(f"📊 Porcentaje de coincidencia: {summary['porcentaje_match_global']}%")
    logger.info("=" * 60)
    
    # Mostrar archivos con más cláusulas no encontradas
    logger.info("🔍 ARCHIVOS CON MÁS CLAÚSULAS NO ENCONTRADAS:")
    sorted_files = sorted(comparison_results.items(), 
                         key=lambda x: x[1]['estadisticas']['unmatched'], 
                         reverse=True)
    
    for file_name, results in sorted_files[:5]:  # Top 5
        unmatched_count = results['estadisticas']['unmatched']
        if unmatched_count > 0:
            logger.info(f"  📄 {file_name}: {unmatched_count} cláusulas no encontradas")
    
    logger.info("=" * 60)
    logger.info(f"💾 Resultados guardados en:")
    logger.info(f"  📄 {output_file}")
    logger.info(f"  📄 {unmatched_file}")

if __name__ == "__main__":
    compare_clauses_with_standard()