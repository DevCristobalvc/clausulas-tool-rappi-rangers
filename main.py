#!/usr/bin/env python3
"""
Script principal para el análisis de cláusulas contractuales.
"""

import sys
import argparse
from pathlib import Path

# Agregar el directorio actual al path
sys.path.append(str(Path(__file__).parent))

from config.settings import ensure_directories, validate_config
from scripts.extract_contract_clauses import process_all_contracts
from scripts.diagnose_pdfs import main as diagnose_main
from scripts.comparar_clausulas import main as compare_main
from scripts.expandir_clausulas_estandar import expand_clausulas_estandar, create_categorized_clauses

def main():
    """Función principal con interfaz de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Sistema de Análisis de Cláusulas Contractuales Rappi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py extract                    # Extraer cláusulas de PDFs
  python main.py diagnose                   # Diagnosticar PDFs problemáticos
  python main.py compare                    # Comparar con cláusulas estándar
  python main.py expand                     # Expandir archivo estándar
  python main.py all                        # Ejecutar todo el flujo
        """
    )
    
    parser.add_argument(
        "command",
        choices=["extract", "diagnose", "compare", "expand", "all"],
        help="Comando a ejecutar"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Modo verbose"
    )
    
    args = parser.parse_args()
    
    # Asegurar que los directorios existan
    ensure_directories()
    
    # Validar configuración
    errors = validate_config()
    if errors:
        print("❌ Errores de configuración:")
        for error in errors:
            print(f"  - {error}")
        return 1
    
    print("🚀 Iniciando análisis de cláusulas contractuales...")
    
    try:
        if args.command == "extract":
            print("📄 Extrayendo cláusulas de PDFs...")
            process_all_contracts()
            
        elif args.command == "diagnose":
            print("🔍 Diagnosticando PDFs...")
            diagnose_main()
            
        elif args.command == "compare":
            print("📊 Comparando con cláusulas estándar...")
            compare_main()
            
        elif args.command == "expand":
            print("📈 Expandiendo archivo estándar...")
            expand_clausulas_estandar()
            create_categorized_clauses()
            
        elif args.command == "all":
            print("🔄 Ejecutando flujo completo...")
            
            # 1. Diagnosticar PDFs
            print("\n1️⃣ Diagnosticando PDFs...")
            diagnose_main()
            
            # 2. Extraer cláusulas
            print("\n2️⃣ Extrayendo cláusulas...")
            process_all_contracts()
            
            # 3. Comparar con estándar
            print("\n3️⃣ Comparando con estándar...")
            compare_main()
            
            # 4. Expandir archivo estándar
            print("\n4️⃣ Expandiendo archivo estándar...")
            expand_clausulas_estandar()
            create_categorized_clauses()
            
            print("\n✅ Flujo completo ejecutado exitosamente!")
        
        print(f"\n🎉 Comando '{args.command}' ejecutado exitosamente!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error ejecutando comando '{args.command}': {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
