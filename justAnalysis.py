from pathlib import Path
import json
from analyze_contract import analyze_contract

# Directorios
INPUT_DIR = Path("results/txtData")
WITHOUT_DATA_DIR = Path("results/withoutData")
WITH_DATA_DIR = Path("results/withData")
WITHOUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
WITH_DATA_DIR.mkdir(parents=True, exist_ok=True)

def default_flat_json():
    # Estructura plana basada en data.md
    return {
        # 1. Desembolsos y Pagos
        "TieneDesembolsos": "NO",
        "PeriodicidadPagos": "",
        "CondicionesPago": "",
        "FormaPago": "",
        "DetalleDesembolsos": "",

        # 2. Exclusividad
        "TieneExclusividad": "NO",
        "AlcanceExclusividad": "NA",
        "CondicionesExclusividad": "",
        "RupturaExclusividad": "NO",
        "DetalleExclusividad": "",

        # 3. Término del Contrato
        "DuracionContrato": "",
        "FechaInicio": "",
        "FechaFin": "",
        "TerminacionUnilateral": "NO",
        "PenalidadTerminacion": "NO",
        "Preaviso": "",
        "DetalleTermino": "",

        # 4. Vigencia y Renovación
        "TieneRenovacionAutomatica": "NO",
        "PeriodicidadRenovacion": "NA",
        "PreavisoNoRenovacion": "NA",
        "DetalleVigencia": ""
    }

def save_json(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    if not INPUT_DIR.exists():
        print(f"⚠️ No existe el directorio de entrada: {INPUT_DIR}")
        return

    txt_files = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() == ".txt"])
    if not txt_files:
        print(f"⚠️ No se encontraron TXT en {INPUT_DIR}")
        return

    for txt_path in txt_files:
        name = txt_path.stem
        try:
            text = txt_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"❌ Error leyendo {txt_path.name}: {e}")
            text = ""

        if not text:
            # Caso SIN DATA: generar JSON plano en results/withoutData
            out_file = WITHOUT_DATA_DIR / f"{name}.json"
            save_json(out_file, default_flat_json())
            print(f"📝 Sin datos → JSON plano guardado en: {out_file}")
            continue

        # Caso CON DATA: analizar usando el pipeline existente
        print(f"🔎 Analizando {txt_path.name} ...")
        result = analyze_contract(text, file_name=name)

        output_data = {
            "archivo": name,
            "analisis": result
        }
        out_file = WITH_DATA_DIR / f"{name}.json"
        save_json(out_file, output_data)
        print(f"✅ Guardado análisis: {out_file}")

if __name__ == "__main__":
    main()