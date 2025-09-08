#!/usr/bin/env python3
"""
Script de demostración de la nueva estructura del proyecto.
"""

import sys
from pathlib import Path

# Agregar el directorio actual al path
sys.path.append(str(Path(__file__).parent))

from config.settings import *
from scripts.utils import print_project_status, validate_data_integrity

def demo_project_structure():
    """Demuestra la nueva estructura del proyecto."""
    print("🎯 DEMOSTRACIÓN DE LA NUEVA ESTRUCTURA DEL PROYECTO")
    print("=" * 60)
    
    print("\n📁 ESTRUCTURA DE DIRECTORIOS:")
    print(f"  📂 Proyecto raíz: {PROJECT_ROOT}")
    print(f"  📂 Datos: {DATA_DIR}")
    print(f"    📁 Raw: {RAW_DATA_DIR}")
    print(f"    📁 Procesados: {PROCESSED_DATA_DIR}")
    print(f"    📁 Salida: {OUTPUT_DATA_DIR}")
    print(f"  📂 Scripts: {SCRIPTS_DIR}")
    print(f"  📂 Configuración: {CONFIG_DIR}")
    print(f"  📂 Documentación: {PROJECT_ROOT / 'docs'}")
    print(f"  📂 Reportes: {REPORTS_DIR}")
    print(f"  📂 Logs: {LOGS_DIR}")
    
    print("\n📋 ARCHIVOS PRINCIPALES:")
    print(f"  📄 Cláusulas estándar: {STANDARD_CLAUSES_FILE}")
    print(f"  📄 Cláusulas expandidas: {EXPANDED_CLAUSES_FILE}")
    print(f"  📄 Cláusulas categorizadas: {CATEGORIZED_CLAUSES_FILE}")
    
    print("\n🔧 CONFIGURACIÓN:")
    print(f"  🔑 API Key configurada: {'✅' if OPENAI_API_KEY else '❌'}")
    print(f"  🌐 Base URL: {OPENAI_BASE_URL}")
    print(f"  🤖 Modelo: {MODEL_NAME}")
    print(f"  📦 Tamaño de lote: {BATCH_SIZE}")
    print(f"  🎯 Umbral de similitud: {SIMILARITY_THRESHOLD}")
    
    print("\n📊 ESTADO ACTUAL:")
    print_project_status()
    
    print("\n🔍 VALIDACIÓN DE INTEGRIDAD:")
    issues = validate_data_integrity()
    if issues:
        print("  ⚠️ Problemas encontrados:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ✅ Todos los archivos están íntegros")
    
    print("\n📚 CATEGORÍAS DE CLAÚSULAS:")
    for key, value in CLAUSE_CATEGORIES.items():
        print(f"  📋 {key}: {value}")
    
    print("\n🚀 COMANDOS DISPONIBLES:")
    print("  python main.py extract    # Extraer cláusulas")
    print("  python main.py diagnose   # Diagnosticar PDFs")
    print("  python main.py compare    # Comparar con estándar")
    print("  python main.py expand     # Expandir archivo estándar")
    print("  python main.py all        # Ejecutar todo el flujo")
    
    print("\n✨ BENEFICIOS DE LA NUEVA ESTRUCTURA:")
    print("  📁 Organización clara por tipo de archivo")
    print("  🔧 Configuración centralizada")
    print("  📊 Reportes organizados")
    print("  🛠️ Scripts modulares y reutilizables")
    print("  📝 Documentación completa")
    print("  🔍 Logs organizados por script")
    print("  ⚙️ Fácil mantenimiento y escalabilidad")

if __name__ == "__main__":
    demo_project_structure()
