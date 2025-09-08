#!/usr/bin/env python3
"""
Script de inicialización del proyecto.
Crea la estructura de directorios y archivos necesarios.
"""

import sys
from pathlib import Path

# Agregar el directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import ensure_directories, validate_config
from scripts.utils import setup_logging, print_project_status, validate_data_integrity

def init_project():
    """Inicializa el proyecto creando la estructura necesaria."""
    logger = setup_logging("init_project")
    
    logger.info("🚀 Inicializando proyecto de análisis de cláusulas...")
    
    try:
        # Crear directorios
        logger.info("📁 Creando estructura de directorios...")
        ensure_directories()
        
        # Validar configuración
        logger.info("⚙️ Validando configuración...")
        errors = validate_config()
        if errors:
            logger.warning("⚠️ Problemas de configuración encontrados:")
            for error in errors:
                logger.warning(f"  - {error}")
        else:
            logger.info("✅ Configuración válida")
        
        # Validar integridad de datos
        logger.info("🔍 Validando integridad de datos...")
        issues = validate_data_integrity()
        if issues:
            logger.warning("⚠️ Problemas de integridad encontrados:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✅ Datos íntegros")
        
        # Mostrar estado del proyecto
        logger.info("📊 Estado actual del proyecto:")
        print_project_status()
        
        logger.info("✅ Proyecto inicializado exitosamente!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error inicializando proyecto: {e}")
        return False

if __name__ == "__main__":
    success = init_project()
    sys.exit(0 if success else 1)
