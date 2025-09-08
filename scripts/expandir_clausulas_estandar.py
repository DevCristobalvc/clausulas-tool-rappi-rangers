#!/usr/bin/env python3
"""
Script para expandir el archivo de cláusulas estándar con las nuevas cláusulas encontradas.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('expansion_clausulas.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def categorize_clause(clause_text: str, title: str, summary: str) -> str:
    """
    Categoriza una cláusula basándose en su contenido, título y resumen.
    """
    text_lower = clause_text.lower()
    title_lower = title.lower() if title else ""
    summary_lower = summary.lower() if summary else ""
    
    # Palabras clave para cada categoría
    categories = {
        "objeto_definiciones": [
            "objeto", "definiciones", "contexto", "capacidad", "partes", "acuerdo"
        ],
        "obligaciones_rappi": [
            "obligaciones de rappi", "rappi se obliga", "rappi deberá", "rappi garantiza"
        ],
        "obligaciones_aliado": [
            "obligaciones del aliado", "aliado se obliga", "aliado deberá", "aliado garantiza"
        ],
        "financiero_pagos": [
            "contraprestación", "comisión", "pago", "reembolso", "bonus", "línea de crédito",
            "fondo", "garantía", "ventas garantizadas", "signing bonus"
        ],
        "operativo_logistica": [
            "entrega", "empaque", "ticket", "preparación", "cooking time", "repartidor",
            "logística", "pick-up", "delivery"
        ],
        "exclusividad_terminacion": [
            "exclusividad", "terminación", "vigencia", "penalidad", "incumplimiento"
        ],
        "especializado": [
            "alcohol", "warrants", "mercadotecnia", "marketing", "permisos", "licencias"
        ],
        "calidad_responsabilidad": [
            "calidad", "higiene", "seguridad", "responsabilidad", "compensación", "reclamo"
        ],
        "confidencialidad_propiedad": [
            "confidencial", "propiedad intelectual", "marca", "software", "información"
        ]
    }
    
    # Buscar la categoría con más coincidencias
    best_category = "general"
    max_matches = 0
    
    for category, keywords in categories.items():
        matches = 0
        for keyword in keywords:
            if keyword in text_lower or keyword in title_lower or keyword in summary_lower:
                matches += 1
        if matches > max_matches:
            max_matches = matches
            best_category = category
    
    return best_category

def expand_clausulas_estandar():
    """
    Expande el archivo de cláusulas estándar con las nuevas cláusulas encontradas.
    """
    logger.info("🔍 EXPANDIENDO CLAÚSULAS ESTÁNDAR")
    
    # Cargar cláusulas estándar existentes
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
    
    # Cargar cláusulas no encontradas
    no_matches_file = Path("salida/clausulas_no_encontradas.json")
    if not no_matches_file.exists():
        logger.error("No se encontró el archivo clausulas_no_encontradas.json")
        return
    
    try:
        with no_matches_file.open("r", encoding="utf-8") as f:
            no_matches = json.load(f)
        logger.info(f"📋 Cláusulas no encontradas cargadas: {len(no_matches)}")
    except Exception as e:
        logger.error(f"Error cargando cláusulas no encontradas: {e}")
        return
    
    # Procesar y categorizar nuevas cláusulas
    new_clauses = []
    categories_count = {}
    
    for clause in no_matches:
        category = categorize_clause(
            clause.get("text", ""),
            clause.get("title", ""),
            clause.get("summary", "")
        )
        
        new_clause = {
            "id": len(standard_clauses) + len(new_clauses) + 1,
            "clausula": clause["text"],
            "titulo": clause.get("title"),
            "resumen": clause.get("summary"),
            "categoria": category,
            "archivo_origen": clause["archivo"],
            "es_nueva": True
        }
        
        new_clauses.append(new_clause)
        categories_count[category] = categories_count.get(category, 0) + 1
    
    # Agregar categoría a cláusulas estándar existentes
    for clause in standard_clauses:
        clause["categoria"] = "estandar_existente"
        clause["es_nueva"] = False
    
    # Combinar cláusulas
    expanded_clauses = standard_clauses + new_clauses
    
    # Guardar archivo expandido
    expanded_file = Path("clausulas_estandar_expandido.json")
    with expanded_file.open("w", encoding="utf-8") as f:
        json.dump(expanded_clauses, f, ensure_ascii=False, indent=2)
    
    # Crear archivo solo con nuevas cláusulas
    new_only_file = Path("clausulas_nuevas_encontradas.json")
    with new_only_file.open("w", encoding="utf-8") as f:
        json.dump(new_clauses, f, ensure_ascii=False, indent=2)
    
    # Crear resumen por categorías
    summary = {
        "total_clausulas_originales": len(standard_clauses),
        "total_clausulas_nuevas": len(new_clauses),
        "total_clausulas_expandidas": len(expanded_clauses),
        "categorias_nuevas": categories_count,
        "archivos_generados": {
            "clausulas_expandidas": str(expanded_file),
            "clausulas_nuevas": str(new_only_file)
        }
    }
    
    summary_file = Path("resumen_expansion_clausulas.json")
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Mostrar resumen
    logger.info("📊 RESUMEN DE EXPANSIÓN:")
    logger.info(f"  📋 Cláusulas originales: {len(standard_clauses)}")
    logger.info(f"  🆕 Cláusulas nuevas: {len(new_clauses)}")
    logger.info(f"  📈 Total expandido: {len(expanded_clauses)}")
    logger.info(f"  📄 Archivo expandido: {expanded_file}")
    logger.info(f"  📄 Solo nuevas: {new_only_file}")
    logger.info(f"  📊 Resumen: {summary_file}")
    
    logger.info("\n📋 DISTRIBUCIÓN POR CATEGORÍAS:")
    for category, count in sorted(categories_count.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {category}: {count} cláusulas")
    
    return expanded_clauses

def create_categorized_clauses():
    """
    Crea un archivo con cláusulas organizadas por categorías.
    """
    logger.info("📂 CREANDO ARCHIVO CATEGORIZADO")
    
    expanded_file = Path("clausulas_estandar_expandido.json")
    if not expanded_file.exists():
        logger.error("No se encontró el archivo clausulas_estandar_expandido.json")
        return
    
    try:
        with expanded_file.open("r", encoding="utf-8") as f:
            clauses = json.load(f)
    except Exception as e:
        logger.error(f"Error cargando cláusulas expandidas: {e}")
        return
    
    # Organizar por categorías
    categorized = {}
    for clause in clauses:
        category = clause.get("categoria", "general")
        if category not in categorized:
            categorized[category] = []
        categorized[category].append(clause)
    
    # Guardar archivo categorizado
    categorized_file = Path("clausulas_por_categoria.json")
    with categorized_file.open("w", encoding="utf-8") as f:
        json.dump(categorized, f, ensure_ascii=False, indent=2)
    
    logger.info(f"📂 Archivo categorizado creado: {categorized_file}")
    
    # Mostrar estadísticas por categoría
    logger.info("\n📊 CLAÚSULAS POR CATEGORÍA:")
    for category, category_clauses in categorized.items():
        logger.info(f"  {category}: {len(category_clauses)} cláusulas")

if __name__ == "__main__":
    expand_clausulas_estandar()
    create_categorized_clauses()
