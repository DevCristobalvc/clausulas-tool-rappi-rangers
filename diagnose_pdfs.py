#!/usr/bin/env python3
"""
Script para diagnosticar PDFs problemáticos y entender por qué no se puede extraer texto.
"""

import os
import json
import logging
from pathlib import Path
import pdfplumber

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def diagnose_pdf_detailed(pdf_path: Path) -> dict:
    """
    Diagnostica un PDF de manera detallada para entender por qué no se puede extraer texto.
    """
    diagnosis = {
        "archivo": pdf_path.name,
        "existe": pdf_path.exists(),
        "tamaño_bytes": 0,
        "es_pdf_valido": False,
        "total_paginas": 0,
        "metodos_exitosos": [],
        "texto_extraido": "",
        "metadatos": {},
        "errores": [],
        "detalles_por_pagina": []
    }
    
    if not pdf_path.exists():
        diagnosis["errores"].append("Archivo no existe")
        return diagnosis
    
    try:
        diagnosis["tamaño_bytes"] = os.path.getsize(pdf_path)
    except Exception as e:
        diagnosis["errores"].append(f"Error obteniendo tamaño: {e}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            diagnosis["es_pdf_valido"] = True
            diagnosis["total_paginas"] = len(pdf.pages)
            
            # Intentar obtener metadatos
            try:
                diagnosis["metadatos"] = pdf.metadata or {}
            except Exception as e:
                diagnosis["errores"].append(f"Error metadatos: {e}")
            
            # Probar diferentes métodos por página
            methods = [
                ("default", lambda p: p.extract_text()),
                ("layout", lambda p: p.extract_text(layout=True)),
                ("text_flow", lambda p: p.extract_text(use_text_flow=True)),
                ("words", lambda p: " ".join([word.get("text", "") for word in p.extract_words()])),
                ("chars", lambda p: " ".join([char.get("text", "") for char in p.chars]))
            ]
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_info = {
                    "numero": page_num,
                    "metodos_exitosos": [],
                    "texto_extraido": "",
                    "errores": []
                }
                
                for method_name, extract_func in methods:
                    try:
                        text = extract_func(page)
                        if text and len(text.strip()) > 10:
                            page_info["metodos_exitosos"].append(method_name)
                            if not page_info["texto_extraido"]:
                                page_info["texto_extraido"] = text[:200] + "..." if len(text) > 200 else text
                    except Exception as e:
                        page_info["errores"].append(f"Método {method_name}: {e}")
                
                diagnosis["detalles_por_pagina"].append(page_info)
                
                # Agregar métodos exitosos al diagnóstico general
                for method in page_info["metodos_exitosos"]:
                    if method not in diagnosis["metodos_exitosos"]:
                        diagnosis["metodos_exitosos"].append(method)
            
            # Intentar extraer texto completo con el mejor método
            if diagnosis["metodos_exitosos"]:
                best_method = diagnosis["metodos_exitosos"][0]
                try:
                    all_text = []
                    for page in pdf.pages:
                        if best_method == "default":
                            text = page.extract_text()
                        elif best_method == "layout":
                            text = page.extract_text(layout=True)
                        elif best_method == "text_flow":
                            text = page.extract_text(use_text_flow=True)
                        elif best_method == "words":
                            text = " ".join([word.get("text", "") for word in page.extract_words()])
                        elif best_method == "chars":
                            text = " ".join([char.get("text", "") for char in page.chars])
                        
                        if text:
                            all_text.append(text)
                    
                    diagnosis["texto_extraido"] = "\n\n".join(all_text)
                    
                except Exception as e:
                    diagnosis["errores"].append(f"Error extrayendo texto completo: {e}")
                
    except Exception as e:
        diagnosis["errores"].append(f"Error abriendo PDF: {e}")
    
    return diagnosis

def main():
    """Función principal para diagnosticar todos los PDFs."""
    input_dir = Path("contratos")
    output_dir = Path("salida")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        logger.error(f"Directorio {input_dir} no existe")
        return
    
    pdf_files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() == ".pdf"])
    if not pdf_files:
        logger.error(f"No se encontraron PDFs en {input_dir}")
        return
    
    logger.info(f"🔍 Diagnosticando {len(pdf_files)} archivos PDF...")
    
    all_diagnoses = []
    problematic_count = 0
    
    for pdf_path in pdf_files:
        logger.info(f"📄 Analizando {pdf_path.name}...")
        diagnosis = diagnose_pdf_detailed(pdf_path)
        all_diagnoses.append(diagnosis)
        
        if not diagnosis["metodos_exitosos"]:
            problematic_count += 1
            logger.warning(f"  ⚠️ PROBLEMÁTICO: {pdf_path.name}")
            logger.warning(f"    - Tamaño: {diagnosis['tamaño_bytes']:,} bytes")
            logger.warning(f"    - Páginas: {diagnosis['total_paginas']}")
            logger.warning(f"    - Errores: {len(diagnosis['errores'])}")
            if diagnosis['errores']:
                for error in diagnosis['errores'][:3]:
                    logger.warning(f"      * {error}")
        else:
            logger.info(f"  ✅ OK: {pdf_path.name} - Métodos exitosos: {', '.join(diagnosis['metodos_exitosos'])}")
            if diagnosis['texto_extraido']:
                logger.info(f"    - Texto extraído: {len(diagnosis['texto_extraido'])} caracteres")
    
    # Guardar diagnóstico completo
    diagnosis_file = output_dir / "diagnostico_completo_pdfs.json"
    with diagnosis_file.open("w", encoding="utf-8") as f:
        json.dump(all_diagnoses, f, ensure_ascii=False, indent=2)
    
    # Generar resumen
    summary = {
        "total_archivos": len(pdf_files),
        "archivos_problematicos": problematic_count,
        "archivos_ok": len(pdf_files) - problematic_count,
        "archivos_problematicos_lista": [d["archivo"] for d in all_diagnoses if not d["metodos_exitosos"]],
        "archivos_ok_lista": [d["archivo"] for d in all_diagnoses if d["metodos_exitosos"]]
    }
    
    summary_file = output_dir / "resumen_diagnostico_pdfs.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info("📊 RESUMEN DEL DIAGNÓSTICO:")
    logger.info(f"  📁 Total archivos: {summary['total_archivos']}")
    logger.info(f"  ✅ Archivos OK: {summary['archivos_ok']}")
    logger.info(f"  ⚠️ Archivos problemáticos: {summary['archivos_problematicos']}")
    logger.info(f"  📄 Diagnóstico completo: {diagnosis_file}")
    logger.info(f"  📋 Resumen: {summary_file}")
    
    if summary['archivos_problematicos'] > 0:
        logger.warning("📋 ARCHIVOS PROBLEMÁTICOS:")
        for archivo in summary['archivos_problematicos_lista']:
            logger.warning(f"  - {archivo}")
        logger.warning("💡 Recomendación: Estos archivos pueden ser PDFs escaneados o tener protección de copia.")

if __name__ == "__main__":
    main()
