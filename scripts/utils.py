"""
Utilidades comunes para el proyecto de análisis de cláusulas.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

# Agregar el directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import *

def setup_logging(script_name: str, log_level: str = LOG_LEVEL) -> logging.Logger:
    """
    Configura el logging para un script específico.
    """
    logger = logging.getLogger(script_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Evitar duplicar handlers
    if logger.handlers:
        return logger
    
    # Crear formatter
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Handler para archivo
    log_file = LOGS_DIR / f"{script_name}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Agregar handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_json_file(file_path: Path) -> Dict[str, Any]:
    """
    Carga un archivo JSON de manera segura.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Error cargando {file_path}: {e}")

def save_json_file(data: Any, file_path: Path, indent: int = 2) -> None:
    """
    Guarda datos en un archivo JSON de manera segura.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
    except Exception as e:
        raise Exception(f"Error guardando {file_path}: {e}")

def get_pdf_files() -> List[Path]:
    """
    Obtiene la lista de archivos PDF en el directorio raw.
    """
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))
    return sorted(pdf_files)

def get_processed_files() -> List[Path]:
    """
    Obtiene la lista de archivos JSON procesados.
    """
    json_files = list(PROCESSED_DATA_DIR.glob("*.json"))
    # Filtrar archivos de diagnóstico
    filtered_files = [
        f for f in json_files 
        if f.name not in [
            "diagnostico_completo_pdfs.json",
            "resumen_diagnostico_pdfs.json",
            "diagnostico_pdfs_problematicos.json",
            "reporte_procesamiento.json"
        ]
    ]
    return sorted(filtered_files)

def print_project_status():
    """
    Imprime el estado actual del proyecto.
    """
    print("📊 ESTADO DEL PROYECTO")
    print(f"  📁 PDFs disponibles: {len(get_pdf_files())}")
    print(f"  📄 Archivos procesados: {len(get_processed_files())}")
    print(f"  📋 Cláusulas estándar: {STANDARD_CLAUSES_FILE.exists()}")
    print(f"  📈 Archivo expandido: {EXPANDED_CLAUSES_FILE.exists()}")
    print(f"  📂 Archivo categorizado: {CATEGORIZED_CLAUSES_FILE.exists()}")

def clean_old_logs():
    """
    Limpia logs antiguos para mantener el directorio ordenado.
    """
    log_files = list(LOGS_DIR.glob("*.log"))
    if len(log_files) > 10:  # Mantener solo los últimos 10 logs
        log_files.sort(key=lambda x: x.stat().st_mtime)
        for old_log in log_files[:-10]:
            old_log.unlink()
            print(f"🗑️ Eliminado log antiguo: {old_log.name}")

def validate_data_integrity():
    """
    Valida la integridad de los datos del proyecto.
    """
    issues = []
    
    # Verificar archivos críticos
    critical_files = [
        STANDARD_CLAUSES_FILE,
        EXPANDED_CLAUSES_FILE,
        CATEGORIZED_CLAUSES_FILE
    ]
    
    for file_path in critical_files:
        if file_path.exists():
            try:
                load_json_file(file_path)
            except Exception as e:
                issues.append(f"Archivo corrupto: {file_path.name} - {e}")
        else:
            issues.append(f"Archivo faltante: {file_path.name}")
    
    # Verificar directorios
    required_dirs = [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DATA_DIR]
    for dir_path in required_dirs:
        if not dir_path.exists():
            issues.append(f"Directorio faltante: {dir_path}")
    
    return issues

def create_backup():
    """
    Crea un backup de los archivos importantes.
    """
    import shutil
    from datetime import datetime
    
    backup_dir = PROJECT_ROOT / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(exist_ok=True)
    
    # Archivos importantes a respaldar
    important_files = [
        STANDARD_CLAUSES_FILE,
        EXPANDED_CLAUSES_FILE,
        CATEGORIZED_CLAUSES_FILE
    ]
    
    for file_path in important_files:
        if file_path.exists():
            shutil.copy2(file_path, backup_dir)
    
    print(f"💾 Backup creado en: {backup_dir}")
    return backup_dir
