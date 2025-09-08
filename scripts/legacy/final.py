import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Definir el alcance de la API
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/spreadsheets",
         "https://www.googleapis.com/auth/drive.file",
         "https://www.googleapis.com/auth/drive"]

# Cargar credenciales
creds = ServiceAccountCredentials.from_json_keyfile_name("final.json", scope)
client = gspread.authorize(creds)

# Abrir la hoja por ID
spreadsheet = client.open_by_key("1evlowTDG_dFKSBAsCFirLZGj9JWIq2MYJ81iUD0Lwag")
sheet = spreadsheet.sheet1  # primera pestaña

# Leer todas las filas
rows = sheet.get_all_values()
for row in rows:
    print(row)

# Escribir en la celda A1
sheet.update_cell(1, 1, "YEESS SR")

# Agregar una fila al final
sheet.append_row(["dato1", "dato2", "dato3"])
print("Datos escritos en la hoja de cálculo.")
