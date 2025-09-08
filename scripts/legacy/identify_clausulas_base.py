import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Cargar .env
load_dotenv()

# Config OpenAI desde variables de entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
clausulas_estandar_FILE = os.path.join(BASE_DIR, "clausulas_estandar.json")
SALIDA_DIR = os.path.join(BASE_DIR, "salida")
IDENTIFY_DIR = os.path.join(BASE_DIR, "identify")

os.makedirs(IDENTIFY_DIR, exist_ok=True)

print("🔎 Cargando cláusulas estándar...")
with open(clausulas_estandar_FILE, "r", encoding="utf-8") as f:
    clausulas_estandar = json.load(f)

clausulas_estandar_texts = [c["clausula"] for c in clausulas_estandar]
print(f"✅ {len(clausulas_estandar_texts)} cláusulas estándar cargadas.\n")


def is_clause_in_base(clause_text: str) -> bool:
    """
    Usa el modelo para determinar si clause_text está contenida en las clausulas_estandar.
    """
    print(f"   🤔 Evaluando cláusula: {clause_text[:80]}...")  # muestra primeras 80 chars
    prompt = f"""
    Tienes una lista de cláusulas base:

    {json.dumps(clausulas_estandar_texts, ensure_ascii=False, indent=2)}

    Pregunta: ¿La siguiente cláusula corresponde (aunque esté redactada diferente) a alguna de las cláusulas base?

    Cláusula: "{clause_text}"

    Responde SOLO con "SI" o "NO".
    """

    #revisar con estructured outputs usando literl pydantic models
    #revisar si usar booleanos
    #con pydantic models agregar un campo par que dé la razón del por qué es o no es cláusula base
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = resp.choices[0].message.content.strip().upper()
    print(f"   ➡️ Respuesta del modelo: {answer}")
    return answer.startswith("SI")


print("🚀 Iniciando identificación de cláusulas extra...\n")

# recorrer archivos en /salida
for file_name in os.listdir(SALIDA_DIR):
    if not file_name.endswith(".json"):
        continue

    print(f"📂 Procesando archivo: {file_name}")
    file_path = os.path.join(SALIDA_DIR, file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        contrato = json.load(f)

    archivo_pdf = contrato.get("archivo", file_name.replace(".json", ".pdf"))
    clausulas = contrato.get("clausulas", [])

    clausulas_extra = []

    print(f"   📑 Total cláusulas detectadas: {len(clausulas)}")

    for idx, c in enumerate(clausulas, start=1):
        text = c.get("text", "").strip()
        title = c.get("title", "Sin título")

        print(f"   🔍 Cláusula {idx}/{len(clausulas)} → {title}")
        if not is_clause_in_base(text):
            clausulas_extra.append(f"{title}: {text}")
            print("   ⚠️  Marcada como EXTRA\n")
        else:
            print("   ✅ Coincide con cláusulas estándar\n")

    salida_identify = {
        "archivo": archivo_pdf,
        "clausulas_extra": clausulas_extra
    }

    out_file = os.path.join(IDENTIFY_DIR, file_name)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(salida_identify, f, ensure_ascii=False, indent=2)

    print(f"📌 Archivo de salida: {out_file}")
    print(f"✅ Procesado {file_name}, encontradas {len(clausulas_extra)} cláusulas extra.\n")

print("🎉 Proceso completado. Revisa la carpeta /identify para los resultados.")
