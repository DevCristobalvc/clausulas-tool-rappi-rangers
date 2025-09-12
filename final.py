import os
import json
import re
import unicodedata
import gspread
import time
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from rapidfuzz import fuzz, process

from dotenv import load_dotenv


# Cargar .env
load_dotenv()

#declarar variables de entorno
spreadsheet_id = os.getenv("SPREADSHEET_ID")

# === CONFIGURACIÓN GOOGLE SHEETS ===
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]

# Cargar credenciales desde tu archivo final.json
creds = ServiceAccountCredentials.from_json_keyfile_name("final.json", scope)
#creds = Credentials.from_authorized_user_file("credentials.json", scope)
client = gspread.authorize(creds)

# Servicio de Google Drive (para buscar archivos)
drive_service = build("drive", "v3", credentials=creds)

# Abrir hoja por ID
spreadsheet = client.open_by_key(spreadsheet_id)
sheet = spreadsheet.sheet1  # primera pestaña

# === CONFIG JSONS DE CONTRATOS SEMANTICOS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_RESULTS_DIR = os.path.join(BASE_DIR, "contratos_semanticos", "json_with_resume")

# Carpeta específica de Drive donde están los PDFs
DRIVE_FOLDER_ID = "1IS9ODgk1SVBK4kbFjglAqrdzgzS1lQbV"

# === CONFIGURACIÓN DE RATE LIMITING ===
SLEEP_BETWEEN_WRITES = 1.5  # segundos entre escrituras
BATCH_SIZE = 10  # procesar en lotes para evitar timeouts

# --- Funciones auxiliares ---
def normalize_filename(name: str) -> str:
    """Normaliza nombres de archivo para comparar de forma robusta."""
    # Quitar acentos raros
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    # Minúsculas
    name = name.lower()
    # Reemplazar espacios múltiples por uno
    name = re.sub(r"\s+", " ", name)
    # Limpiar doble extensión
    name = re.sub(r"\.pdf\.pdf$", ".pdf", name)
    return name.strip()


def obtener_archivos_existentes() -> set:
    """Obtiene la lista de archivos ya procesados en la hoja."""
    try:
        print("[INFO] Verificando archivos existentes en la hoja...")
        time.sleep(0.5)  # pequeño delay para la lectura
        
        # Obtener todos los valores de la primera columna (excluyendo header)
        valores_columna_a = sheet.col_values(1)[1:]  # [1:] para saltar el header
        
        # Normalizar nombres para comparación robusta
        archivos_existentes = {normalize_filename(nombre) for nombre in valores_columna_a if nombre.strip()}
        
        print(f"[INFO] Se encontraron {len(archivos_existentes)} archivos ya procesados")
        return archivos_existentes
        
    except Exception as e:
        print(f"[WARN] Error obteniendo archivos existentes: {e}")
        return set()


def buscar_link_en_drive(nombre_pdf: str) -> str:
    """Busca un archivo PDF en Drive usando normalización + fuzzy matching."""
    try:
        query = (
            f"mimeType='application/pdf' "
            f"and '{DRIVE_FOLDER_ID}' in parents"
        )
        results = drive_service.files().list(
            q=query,
            spaces="drive",
            fields="files(id, name)",
            pageSize=1000  # traemos hasta 1000 archivos de la carpeta
        ).execute()

        items = results.get("files", [])
        if not items:
            print(f"[WARN] No se encontraron PDFs en la carpeta Drive para {nombre_pdf}")
            return ""

        # Normalizar todos los nombres de drive
        drive_files = {normalize_filename(item["name"]): item for item in items}

        # Normalizar el query
        query_norm = normalize_filename(nombre_pdf)

        # Si hay match exacto tras normalización
        if query_norm in drive_files:
            file_id = drive_files[query_norm]["id"]
            return f"https://drive.google.com/file/d/{file_id}/view"

        # Si no hay match exacto, usamos fuzzy matching
        best_match = process.extractOne(
            query_norm,
            drive_files.keys(),
            scorer=fuzz.token_sort_ratio
        )

        if best_match and best_match[1] > 85:  # umbral de similitud
            matched_name = best_match[0]
            file_id = drive_files[matched_name]["id"]
            print(f"[WARN] No coincidió exactamente: {nombre_pdf}, mejor match: {drive_files[matched_name]['name']} (score {best_match[1]})")
            return f"https://drive.google.com/file/d/{file_id}/view"

        print(f"[WARN] No se encontró coincidencia para {nombre_pdf}")
        return ""

    except Exception as e:
        print(f"[WARN] Error buscando {nombre_pdf} en Drive: {e}")
        return ""


def obtener_estructura_columnas(json_dir: str) -> list:
    """Obtiene todas las columnas necesarias analizando la estructura de los JSONs."""
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    estructura = {}
    
    print(f"[DEBUG] Analizando estructura de {len(json_files)} archivos JSON...")
    
    # Analizar cada archivo JSON
    for file_name in json_files:
        file_path = os.path.join(json_dir, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Analizar cada clave del JSON
            for key, value in data.items():
                if key.startswith("_"):  # Excluir debug
                    continue
                
                if isinstance(value, dict):
                    # Es una sección con subclaves
                    if key not in estructura:
                        estructura[key] = set()
                    estructura[key].update(value.keys())
                else:
                    # Es un valor directo
                    estructura[key] = "simple"
            
        except Exception as e:
            print(f"[ERROR] Error procesando {file_name}: {e}")
    
    # Crear lista de columnas
    columnas = []
    
    # Agregar claves simples primero
    for key, subclaves in estructura.items():
        if subclaves == "simple":
            columnas.append(key)
            print(f"[DEBUG] Columna simple: {key}")
    
    # Agregar claves compuestas
    for key, subclaves in estructura.items():
        if subclaves != "simple":
            for subclave in sorted(subclaves):
                columna_completa = f"{key}.{subclave}"
                columnas.append(columna_completa)
                print(f"[DEBUG] Columna compuesta: {columna_completa}")
    
    print(f"[DEBUG] Total columnas: {len(columnas)}")
    return columnas


def convertir_valor_a_texto(valor) -> str:
    """Convierte cualquier valor a texto para insertar en la celda."""
    if isinstance(valor, dict):
        return json.dumps(valor, ensure_ascii=False, indent=2)
    elif isinstance(valor, list):
        return json.dumps(valor, ensure_ascii=False)
    elif valor is None:
        return ""
    else:
        return str(valor)


def crear_header_si_no_existe(columnas: list) -> bool:
    """Crea el header si la hoja está vacía, retorna True si se creó."""
    try:
        # Verificar si la hoja tiene contenido
        primer_valor = sheet.acell('A1').value
        
        if not primer_valor or primer_valor.strip() == "":
            # La hoja está vacía, crear header
            header = ["Archivo", "Link"] + columnas
            sheet.append_row(header)
            time.sleep(SLEEP_BETWEEN_WRITES)
            print("[OK] Header creado en la hoja de cálculo.")
            return True
        else:
            print("[INFO] Header ya existe en la hoja.")
            return False
            
    except Exception as e:
        print(f"[WARN] Error verificando header: {e}")
        return False


def safe_append_row(fila: list, archivo_nombre: str) -> bool:
    """Intenta escribir una fila con manejo de errores y retry."""
    max_retries = 3
    for intento in range(max_retries):
        try:
            sheet.append_row(fila)
            time.sleep(SLEEP_BETWEEN_WRITES)  # Sleep después de cada escritura
            print(f"[OK] Agregada fila para {archivo_nombre}")
            return True
            
        except Exception as e:
            if "429" in str(e) or "Quota exceeded" in str(e):
                wait_time = (intento + 1) * 5  # 5, 10, 15 segundos
                print(f"[WARN] Rate limit excedido para {archivo_nombre}. Esperando {wait_time}s... (intento {intento + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"[ERROR] Error escribiendo {archivo_nombre}: {e}")
                return False
    
    print(f"[ERROR] No se pudo escribir {archivo_nombre} después de {max_retries} intentos")
    return False


# === VERIFICAR CARPETA ===
if not os.path.exists(JSON_RESULTS_DIR):
    print(f"[ERROR] No existe la carpeta: {JSON_RESULTS_DIR}")
    exit(1)

# === PASO 1: Obtener estructura de columnas ===
columnas = obtener_estructura_columnas(JSON_RESULTS_DIR)
print(f"[INFO] Se crearon {len(columnas)} columnas")

# === PASO 2: Crear header si no existe ===
crear_header_si_no_existe(columnas)

# === PASO 3: Obtener archivos ya procesados ===
archivos_existentes = obtener_archivos_existentes()

# === PASO 4: Procesar cada JSON ===
json_files = [f for f in os.listdir(JSON_RESULTS_DIR) if f.endswith(".json")]
archivos_procesados = 0
archivos_saltados = 0

print(f"[INFO] Iniciando procesamiento de {len(json_files)} archivos JSON...")

for i, file_name in enumerate(json_files):
    file_path = os.path.join(JSON_RESULTS_DIR, file_name)
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Obtener nombre del archivo original
        archivo_original = data.get("archivo", file_name.replace(".json", ".pdf"))
        
        # Convertir .txt a .pdf si es necesario
        if archivo_original.endswith(".txt"):
            archivo_original = archivo_original.replace(".txt", ".pdf")

        # VALIDACIÓN: Verificar si ya existe
        archivo_normalizado = normalize_filename(archivo_original)
        if archivo_normalizado in archivos_existentes:
            archivos_saltados += 1
            print(f"[SKIP] {archivo_original} ya existe en la hoja. Saltando...")
            continue

        # Buscar el link en Drive
        archivo_link = buscar_link_en_drive(archivo_original)

        # Crear diccionario de valores disponibles
        valores_disponibles = {}
        
        for key, value in data.items():
            if key.startswith("_"):
                continue
            
            if isinstance(value, dict):
                # Sección con subvalores
                for subclave, subvalor in value.items():
                    clave_completa = f"{key}.{subclave}"
                    valores_disponibles[clave_completa] = convertir_valor_a_texto(subvalor)
            else:
                # Valor simple
                valores_disponibles[key] = convertir_valor_a_texto(value)
        
        # Crear fila con valores
        fila = [archivo_original, archivo_link]
        
        for columna in columnas:
            valor = valores_disponibles.get(columna, "")
            fila.append(valor)

        # Escribir fila con manejo seguro
        if safe_append_row(fila, archivo_original):
            archivos_procesados += 1
            # Agregar a la lista de existentes para evitar duplicados en la misma ejecución
            archivos_existentes.add(archivo_normalizado)
        
        # Mostrar progreso cada 10 archivos
        if (i + 1) % 10 == 0:
            print(f"[PROGRESS] Procesados {i + 1}/{len(json_files)} archivos. Nuevos: {archivos_procesados}, Saltados: {archivos_saltados}")
        
    except Exception as e:
        print(f"[ERROR] Error procesando {file_name}: {e}")
        continue

print(f"\n[RESUMEN] Proceso completado:")
print(f"  - Archivos JSON encontrados: {len(json_files)}")
print(f"  - Archivos nuevos procesados: {archivos_procesados}")
print(f"  - Archivos saltados (ya existían): {archivos_saltados}")
print(f"  - Errores: {len(json_files) - archivos_procesados - archivos_saltados}")