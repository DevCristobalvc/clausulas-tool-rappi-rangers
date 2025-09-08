import os
import json
import re
import unicodedata
import gspread
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
client = gspread.authorize(creds)

# Servicio de Google Drive (para buscar archivos)
drive_service = build("drive", "v3", credentials=creds)

# Abrir hoja por ID
spreadsheet = client.open_by_key(spreadsheet_id)
sheet = spreadsheet.sheet1  # primera pestaña

# === CONFIG JSONS DE IDENTIFY ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IDENTIFY_DIR = os.path.join(BASE_DIR, "identify")

# Carpeta específica de Drive donde están los PDFs
DRIVE_FOLDER_ID = "1IS9ODgk1SVBK4kbFjglAqrdzgzS1lQbV"

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


# === PASO 1: Calcular el máximo de cláusulas ===
max_clausulas = 0
json_files = [f for f in os.listdir(IDENTIFY_DIR) if f.endswith(".json")]

for file_name in json_files:
    file_path = os.path.join(IDENTIFY_DIR, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    clausulas_extra = data.get("clausulas_extra", [])
    if len(clausulas_extra) > max_clausulas:
        max_clausulas = len(clausulas_extra)

print(f"[INFO] Máximo de cláusulas detectadas: {max_clausulas}")

# === PASO 2: Crear cabecera dinámica ===
sheet.clear()
header = ["Archivo", "Link"] + [f"Cláusula {i+1}" for i in range(max_clausulas)]
sheet.append_row(header)
print("[OK] Cabecera creada en la hoja de cálculo.")

# === PASO 3: Escribir filas con datos ===
for file_name in json_files:
    file_path = os.path.join(IDENTIFY_DIR, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    archivo = data.get("archivo", file_name.replace(".json", ".pdf"))
    clausulas_extra = data.get("clausulas_extra", [])

    # Buscar el link en la carpeta de Drive con fuzzy matching
    archivo_link = buscar_link_en_drive(archivo)

    # Ajustar el tamaño de la fila para que coincida con la cabecera
    fila = [archivo, archivo_link] + clausulas_extra
    while len(fila) < len(header):
        fila.append("")  # completar con celdas vacías si faltan

    sheet.append_row(fila)
    print(f"[OK] Agregada fila para {archivo} con {len(clausulas_extra)} cláusulas.")
