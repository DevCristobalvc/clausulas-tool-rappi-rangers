"""
Configuración centralizada para el proyecto de análisis de cláusulas contractuales.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIG_DIR = PROJECT_ROOT / "config"

# Rutas de datos
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DATA_DIR = DATA_DIR / "output"

# Archivos específicos
STANDARD_CLAUSES_FILE = OUTPUT_DATA_DIR / "clausulas_estandar.json"
EXPANDED_CLAUSES_FILE = OUTPUT_DATA_DIR / "clausulas_estandar_expandido.json"
CATEGORIZED_CLAUSES_FILE = OUTPUT_DATA_DIR / "clausulas_por_categoria.json"

# Configuración de OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://rappi.litellm-prod.ai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5-mini")

# Configuración de procesamiento
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
MIN_PARAGRAPH_LENGTH = int(os.getenv("MIN_PARAGRAPH_LENGTH", "50"))

# Configuración de logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Categorías de cláusulas
CLAUSE_CATEGORIES = {
    "objeto_definiciones": "Objeto y Definiciones",
    "obligaciones_rappi": "Obligaciones de Rappi",
    "obligaciones_aliado": "Obligaciones del Aliado",
    "financiero_pagos": "Financiero y Pagos",
    "operativo_logistica": "Operativo y Logística",
    "exclusividad_terminacion": "Exclusividad y Terminación",
    "calidad_responsabilidad": "Calidad y Responsabilidad",
    "confidencialidad_propiedad": "Confidencialidad y Propiedad",
    "especializado": "Especializado",
    "general": "General"
}

def validate_config():
    """Valida la configuración del proyecto."""
    errors = []
    
    # Verificar variables de entorno críticas
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY no está configurada")
    
    if not OPENAI_BASE_URL:
        errors.append("OPENAI_BASE_URL no está configurada")
    
    # Verificar directorios
    required_dirs = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DATA_DIR,
        SCRIPTS_DIR, LOGS_DIR, REPORTS_DIR, CONFIG_DIR
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            errors.append(f"Directorio requerido no existe: {dir_path}")
    
    return errors

def get_pdf_files():
    """Obtiene la lista de archivos PDF en el directorio raw."""
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))
    return sorted(pdf_files)

def get_processed_files():
    """Obtiene la lista de archivos JSON procesados."""
    json_files = list(PROCESSED_DATA_DIR.glob("*.json"))
    return sorted(json_files)

def ensure_directories():
    """Asegura que todos los directorios necesarios existan."""
    required_dirs = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DATA_DIR,
        SCRIPTS_DIR, LOGS_DIR, REPORTS_DIR, CONFIG_DIR
    ]
    
    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
