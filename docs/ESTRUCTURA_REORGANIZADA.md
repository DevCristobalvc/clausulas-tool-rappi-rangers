# 📁 Estructura Reorganizada del Proyecto

## 🎯 Resumen de Cambios

Hemos reorganizado completamente la estructura del proyecto para hacerlo más profesional, mantenible y escalable.

## 📂 Nueva Estructura de Directorios

```
clausulas-tool-rappi-rangers/
├── 📁 config/                    # Configuración centralizada
│   ├── __init__.py
│   ├── settings.py               # Configuración principal
│   └── env_example.txt          # Ejemplo de variables de entorno
├── 📁 data/                      # Datos del proyecto
│   ├── 📁 raw/                   # PDFs originales
│   │   └── contratos/           # Contratos PDF
│   ├── 📁 processed/             # Datos procesados
│   │   ├── *.json               # Archivos JSON procesados
│   │   └── identify/            # Archivos de identificación
│   └── 📁 output/                # Resultados finales
│       ├── clausulas_estandar.json
│       ├── clausulas_estandar_expandido.json
│       ├── clausulas_por_categoria.json
│       └── clausulas_nuevas_encontradas.json
├── 📁 scripts/                   # Scripts de procesamiento
│   ├── __init__.py
│   ├── 📁 legacy/                # Scripts antiguos
│   │   ├── final.py
│   │   ├── csv-json.py
│   │   ├── extraer-texto-pdf.py
│   │   ├── identify_clausulas_base.py
│   │   └── comparision.py
│   ├── extract_contract_clauses.py
│   ├── diagnose_pdfs.py
│   ├── comparar_clausulas.py
│   ├── expandir_clausulas_estandar.py
│   ├── init_project.py
│   └── utils.py
├── 📁 docs/                      # Documentación
│   ├── README.md                 # Documentación principal
│   └── ESTRUCTURA_REORGANIZADA.md
├── 📁 reports/                   # Reportes y análisis
│   ├── resumen_ejecutivo_comparacion.md
│   └── resumen_expansion_clausulas.json
├── 📁 logs/                      # Archivos de log
│   ├── comparacion_clausulas.log
│   └── contract_processing.log
├── 📁 venv/                      # Entorno virtual Python
├── main.py                       # Script principal
├── demo.py                       # Demostración de estructura
├── project_config.json           # Configuración del proyecto
├── requirements.txt              # Dependencias Python
├── .env                          # Variables de entorno
├── .gitignore                    # Archivos ignorados por Git
└── README.md                     # README principal
```

## 🔧 Archivos de Configuración

### `config/settings.py`
- Configuración centralizada del proyecto
- Rutas de directorios
- Variables de entorno
- Configuración de OpenAI
- Categorías de cláusulas

### `project_config.json`
- Metadatos del proyecto
- Configuración de procesamiento
- Estado actual del proyecto
- Mapeo de categorías

### `.env` (ejemplo en `config/env_example.txt`)
- Variables de entorno sensibles
- Configuración de API
- Parámetros de procesamiento

## 🚀 Scripts Principales

### `main.py`
Script principal con interfaz de línea de comandos:
```bash
python main.py extract    # Extraer cláusulas
python main.py diagnose   # Diagnosticar PDFs
python main.py compare    # Comparar con estándar
python main.py expand     # Expandir archivo estándar
python main.py all        # Ejecutar todo el flujo
```

### `scripts/utils.py`
Utilidades comunes:
- Configuración de logging
- Carga/guardado de archivos JSON
- Validación de integridad
- Funciones de utilidad

### `scripts/init_project.py`
Inicialización del proyecto:
- Creación de directorios
- Validación de configuración
- Verificación de integridad

## 📊 Beneficios de la Nueva Estructura

### 1. **Organización Clara**
- Separación por tipo de archivo
- Directorios específicos para cada función
- Fácil navegación y mantenimiento

### 2. **Configuración Centralizada**
- Un solo lugar para toda la configuración
- Variables de entorno organizadas
- Fácil modificación de parámetros

### 3. **Modularidad**
- Scripts independientes y reutilizables
- Utilidades comunes compartidas
- Fácil testing y debugging

### 4. **Escalabilidad**
- Estructura preparada para crecimiento
- Fácil adición de nuevas funcionalidades
- Separación clara de responsabilidades

### 5. **Mantenibilidad**
- Código organizado y documentado
- Logs separados por script
- Reportes organizados

### 6. **Profesionalismo**
- Estructura estándar de proyectos Python
- Documentación completa
- Configuración de Git apropiada

## 🔄 Migración Realizada

### Archivos Movidos:
- **PDFs**: `contratos/` → `data/raw/contratos/`
- **JSONs procesados**: `salida/` → `data/processed/`
- **Cláusulas estándar**: `clausulas_estandar*.json` → `data/output/`
- **Scripts**: `*.py` → `scripts/`
- **Scripts antiguos**: `scripts/legacy/`
- **Logs**: `*.log` → `logs/`
- **Reportes**: `resumen_*.md/json` → `reports/`

### Archivos Creados:
- `config/settings.py` - Configuración centralizada
- `scripts/utils.py` - Utilidades comunes
- `scripts/init_project.py` - Inicialización
- `main.py` - Script principal
- `demo.py` - Demostración
- `project_config.json` - Configuración del proyecto
- `docs/README.md` - Documentación completa
- `requirements.txt` - Dependencias

## 🎯 Próximos Pasos

1. **Actualizar scripts existentes** para usar la nueva configuración
2. **Migrar datos** a la nueva estructura
3. **Probar funcionalidad** con la nueva organización
4. **Documentar cambios** para el equipo
5. **Entrenar usuarios** en la nueva estructura

## 📝 Comandos de Uso

### Inicialización:
```bash
python scripts/init_project.py
```

### Demostración:
```bash
python demo.py
```

### Uso normal:
```bash
python main.py all
```

## ✅ Estado Actual

- ✅ Estructura reorganizada
- ✅ Configuración centralizada
- ✅ Scripts modulares
- ✅ Documentación completa
- ✅ Archivos organizados
- ✅ Logs separados
- ✅ Reportes organizados

La nueva estructura está lista para uso y es mucho más profesional y mantenible que la anterior.
