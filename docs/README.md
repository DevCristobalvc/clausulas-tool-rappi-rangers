# 📋 Sistema de Análisis de Cláusulas Contractuales Rappi

## 🎯 Descripción

Sistema automatizado para extraer, analizar y comparar cláusulas contractuales de documentos PDF relacionados con acuerdos comerciales de Rappi.

## 📁 Estructura del Proyecto

```
clausulas-tool-rappi-rangers/
├── 📁 config/                    # Configuración del proyecto
│   ├── settings.py               # Configuración centralizada
│   └── env_example.txt          # Ejemplo de variables de entorno
├── 📁 data/                      # Datos del proyecto
│   ├── 📁 raw/                   # PDFs originales
│   ├── 📁 processed/             # Datos procesados (JSON)
│   └── 📁 output/                # Resultados finales
├── 📁 scripts/                   # Scripts de procesamiento
│   ├── extract_contract_clauses.py
│   ├── diagnose_pdfs.py
│   ├── comparar_clausulas.py
│   └── expandir_clausulas_estandar.py
├── 📁 docs/                      # Documentación
│   └── README.md                 # Este archivo
├── 📁 reports/                   # Reportes y análisis
├── 📁 logs/                      # Archivos de log
└── 📁 venv/                      # Entorno virtual Python
```

## 🚀 Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd clausulas-tool-rappi-rangers
```

### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno
```bash
cp config/env_example.txt .env
# Editar .env con tus credenciales
```

## 📊 Flujo de Trabajo

### 1. Extracción de Cláusulas
```bash
python scripts/extract_contract_clauses.py
```
- Procesa PDFs en `data/raw/`
- Extrae texto y identifica cláusulas
- Guarda resultados en `data/processed/`

### 2. Diagnóstico de PDFs
```bash
python scripts/diagnose_pdfs.py
```
- Identifica PDFs problemáticos
- Genera reportes de diagnóstico

### 3. Comparación con Estándar
```bash
python scripts/comparar_clausulas.py
```
- Compara cláusulas extraídas con estándar
- Identifica cláusulas nuevas
- Genera reportes de cobertura

### 4. Expansión del Estándar
```bash
python scripts/expandir_clausulas_estandar.py
```
- Expande archivo de cláusulas estándar
- Categoriza nuevas cláusulas
- Genera archivos organizados

## 📋 Archivos de Datos

### Archivos de Entrada
- **PDFs**: `data/raw/contratos/` - Contratos originales
- **Cláusulas Estándar**: `data/output/clausulas_estandar.json`

### Archivos de Salida
- **Procesados**: `data/processed/` - JSON con cláusulas extraídas
- **Expandido**: `data/output/clausulas_estandar_expandido.json`
- **Categorizado**: `data/output/clausulas_por_categoria.json`
- **Reportes**: `reports/` - Análisis y estadísticas

## 🔧 Configuración

### Variables de Entorno
- `OPENAI_API_KEY`: Clave API de OpenAI
- `OPENAI_BASE_URL`: URL base del servicio
- `MODEL_NAME`: Modelo a utilizar
- `BATCH_SIZE`: Tamaño de lote para procesamiento
- `SIMILARITY_THRESHOLD`: Umbral de similitud para comparación

### Categorías de Cláusulas
1. **Objeto y Definiciones** - Contexto y alcance
2. **Obligaciones de Rappi** - Servicios de plataforma
3. **Obligaciones del Aliado** - Cumplimiento operativo
4. **Financiero y Pagos** - Comisiones, bonos, créditos
5. **Operativo y Logística** - Entrega, empaque, tickets
6. **Exclusividad y Terminación** - Penalidades, vigencia
7. **Calidad y Responsabilidad** - Higiene, seguridad
8. **Confidencialidad y Propiedad** - IP, marca, software
9. **Especializado** - Alcohol, warrants, permisos
10. **General** - No categorizada específicamente

## 📈 Resultados Actuales

- **Total cláusulas estándar**: 57
- **Cláusulas nuevas encontradas**: 74
- **Total cláusulas expandidas**: 131
- **Archivos procesados**: 7 PDFs exitosos
- **Cobertura inicial**: 1.3% (indicando necesidad de expansión)

## 🛠️ Scripts Disponibles

### `extract_contract_clauses.py`
- Extrae cláusulas de PDFs
- Utiliza LLM para identificación
- Genera JSON estructurado

### `diagnose_pdfs.py`
- Diagnostica PDFs problemáticos
- Identifica archivos escaneados
- Genera reportes de diagnóstico

### `comparar_clausulas.py`
- Compara con cláusulas estándar
- Calcula similitud de texto
- Identifica cláusulas nuevas

### `expandir_clausulas_estandar.py`
- Expande archivo estándar
- Categoriza automáticamente
- Organiza por tipos

## 📊 Reportes Generados

- **Resumen Ejecutivo**: `reports/resumen_ejecutivo_comparacion.md`
- **Estadísticas de Expansión**: `reports/resumen_expansion_clausulas.json`
- **Logs de Procesamiento**: `logs/`

## 🔍 Troubleshooting

### PDFs No Procesados
- Verificar que no sean PDFs escaneados
- Usar `diagnose_pdfs.py` para diagnóstico
- Considerar OCR para PDFs de imagen

### Baja Cobertura
- Normal en contratos reales vs estándar
- Indica necesidad de expansión del estándar
- Revisar algoritmo de similitud

### Errores de API
- Verificar credenciales en `.env`
- Revisar límites de API
- Verificar conectividad

## 🤝 Contribución

1. Fork el repositorio
2. Crear rama de feature
3. Hacer cambios
4. Crear Pull Request

## 📄 Licencia

[Especificar licencia]

## 📞 Contacto

[Información de contacto]