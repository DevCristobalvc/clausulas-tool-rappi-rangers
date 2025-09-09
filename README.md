# clausulas-tool-rappi-rangers

Herramienta automatizada para la extracción, análisis y reporte de cláusulas contractuales desde archivos PDF, integrando resultados en Google Sheets.

## Descripción

Este proyecto permite:
- Extraer texto de contratos en PDF.
- Analizar el contenido usando modelos de lenguaje (LLM).
- Identificar cláusulas extra respecto a una base estándar.
- Generar reportes estructurados en Google Sheets, incluyendo enlaces a los archivos originales en Google Drive.

## Estructura de carpetas

```
data/
  pdfs/        # Contratos en PDF
  txts/        # Texto extraído de los PDFs
  outputs/     # Resultados del análisis en JSON
identify/      # Resultados de identificación de cláusulas extra
final.py       # Script para reporte en Google Sheets
main.py        # Pipeline principal de extracción y análisis
extract_text.py
analyze_contract.py
config.py
models.py
utils.py
.env           # Variables de entorno (no versionar)
final.json     # Credenciales Google (no versionar)
requirements.txt
```

## Instalación

1. **Clona el repositorio:**
   ```bash
   git clone <URL_DEL_REPO>
   cd clausulas-tool-rappi-rangers
   ```

2. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configura las variables de entorno:**
   - Crea un archivo `.env` con tus claves de OpenAI y Google Sheets.
   - Agrega el archivo de credenciales `final.json` para Google API.

## Ejecución

1. **Extraer texto y analizar contratos:**
   ```bash
   python main.py
   ```

2. **Identificar cláusulas extra (opcional):**
   ```bash
   python identify_clausulas_base.py
   ```

3. **Generar reporte en Google Sheets:**
   ```bash
   python final.py
   ```

## Variables de entorno requeridas (`.env`)

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `MODEL_NAME`
- `SPREADSHEET_ID`
- `SHEET_NAME`

## Dependencias principales

- pdfplumber
- openai
- pydantic
- python-dotenv
- gspread
- oauth2client
- google-api-python-client
- rapidfuzz

## Notas

- Los archivos `.env` y `final.json` están excluidos del control de versiones por seguridad.
- El sistema requiere acceso a Google Drive y Sheets mediante credenciales de servicio.

## Licencia

Este proyecto es privado y para uso interno de Rappi Rangers.