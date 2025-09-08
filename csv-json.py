import csv
import json

def csv_a_json(archivo_csv, archivo_json):
    datos = []
    
    with open(archivo_csv, 'r', encoding='utf-8') as csvfile:
        lector = csv.DictReader(csvfile)
        for fila in lector:
            datos.append({
                "id": int(fila['ID']),
                "clausula": fila['CLÁUSULA'].strip()
            })
    
    with open(archivo_json, 'w', encoding='utf-8') as jsonfile:
        json.dump(datos, jsonfile, ensure_ascii=False, indent=2)
    
    return datos

# Usar la función
archivo_csv = 'Automatización-Clausulas-Rappi-Rangers-Tools.csv'  # Cambia por el nombre de tu archivo
archivo_json = 'clausulas.json'

resultado = csv_a_json(archivo_csv, archivo_json)
print("JSON creado exitosamente!")
print(f"Se procesaron {len(resultado)} cláusulas")