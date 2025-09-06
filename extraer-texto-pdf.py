import os
import pdfplumber
from tkinter import Tk, filedialog

def extraer_texto_pdf(ruta_pdf):
    texto_completo = ""
    with pdfplumber.open(ruta_pdf) as pdf:
        for i, pagina in enumerate(pdf.pages, start=1):
            texto = pagina.extract_text()
            if texto:
                texto_completo += f"\n--- Página {i} ---\n"
                texto_completo += texto
    return texto_completo

if __name__ == "__main__":
    # Ocultar ventana raíz de Tkinter
    Tk().withdraw()
    
    # Abrir explorador de archivos
    ruta_pdf = filedialog.askopenfilename(
        title="Selecciona un archivo PDF",
        filetypes=[("Archivos PDF", "*.pdf")]
    )
    
    if ruta_pdf:
        # Extraer texto
        texto = extraer_texto_pdf(ruta_pdf)
        
        # Crear carpeta "salida" si no existe
        carpeta_salida = "salida"
        os.makedirs(carpeta_salida, exist_ok=True)
        
        # Nombre del archivo de salida
        nombre_pdf = os.path.splitext(os.path.basename(ruta_pdf))[0]
        archivo_txt = os.path.join(carpeta_salida, f"{nombre_pdf}.txt")
        
        # Guardar en archivo de texto
        with open(archivo_txt, "w", encoding="utf-8") as f:
            f.write(texto)
        
        print(f"✅ Texto extraído y guardado en: {archivo_txt}")
    else:
        print("❌ No seleccionaste ningún archivo.")
