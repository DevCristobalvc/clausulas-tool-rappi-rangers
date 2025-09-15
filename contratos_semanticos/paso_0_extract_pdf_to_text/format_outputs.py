#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para formatear archivos de texto de contratos en la carpeta outputs/
Estandariza el formato de los documentos legales extraídos de PDFs.
"""

import os
import re
import glob
from pathlib import Path

def clean_text(text):
    """
    Limpia y formatea el texto de un contrato legal.
    
    Args:
        text (str): Texto original del contrato
        
    Returns:
        str: Texto formateado y limpio
    """
    # Eliminar caracteres de control y espacios en blanco problemáticos
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalizar espacios en blanco múltiples
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalizar saltos de línea múltiples (máximo 2 consecutivos)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Limpiar espacios al inicio y final de líneas
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Limpiar espacios al inicio y final
        line = line.strip()
        
        # Si la línea está vacía, mantenerla como está
        if not line:
            cleaned_lines.append('')
            continue
            
        # Si la línea es muy corta (menos de 3 caracteres) y no es un número romano o letra
        if len(line) < 3 and not re.match(r'^[IVX]+\.?$|^[A-Z]\.?$|^\d+\.?$', line):
            # Unir con la línea siguiente si existe
            if cleaned_lines and cleaned_lines[-1]:
                cleaned_lines[-1] += ' ' + line
            else:
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    
    # Reconstruir el texto
    text = '\n'.join(cleaned_lines)
    
    # Formatear párrafos de declaraciones legales
    text = format_legal_paragraphs(text)
    
    # Limpiar espacios en blanco finales
    text = text.strip()
    
    return text

def format_legal_paragraphs(text):
    """
    Formatea párrafos específicos de documentos legales.
    
    Args:
        text (str): Texto del contrato
        
    Returns:
        str: Texto con párrafos formateados
    """
    # Patrones comunes en contratos legales
    patterns = [
        # Declaraciones con números romanos
        (r'([IVX]+\.)\s*\n\s*([A-Z])', r'\1 \2'),
        # Letras con puntos
        (r'([A-Z]\.)\s*\n\s*([A-Z])', r'\1 \2'),
        # Números con puntos
        (r'(\d+\.)\s*\n\s*([A-Z])', r'\1 \2'),
        # Cláusulas
        (r'(CLÁUSULA\s+[A-Z\s]+\.)\s*\n\s*([A-Z])', r'\1 \2'),
        # Artículos
        (r'(ARTÍCULO\s+\d+\.)\s*\n\s*([A-Z])', r'\1 \2'),
        # Definiciones
        (r'([A-Z][A-Z\s]+\.)\s*\n\s*([A-Z])', r'\1 \2'),
    ]
    
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    
    return text

def format_contract_file(input_path, output_path=None):
    """
    Formatea un archivo de contrato individual.
    
    Args:
        input_path (str): Ruta del archivo de entrada
        output_path (str, optional): Ruta del archivo de salida. Si no se especifica,
                                   se sobrescribe el archivo original.
    """
    try:
        # Leer el archivo original
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            original_text = f.read()
        
        # Limpiar y formatear el texto
        formatted_text = clean_text(original_text)
        
        # Determinar la ruta de salida
        if output_path is None:
            output_path = input_path
        
        # Escribir el archivo formateado
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_text)
        
        print(f"✓ Formateado: {os.path.basename(input_path)}")
        return True
        
    except Exception as e:
        print(f"✗ Error procesando {os.path.basename(input_path)}: {str(e)}")
        return False

def format_all_outputs(outputs_dir, backup=True):
    """
    Formatea todos los archivos .txt en la carpeta outputs.
    
    Args:
        outputs_dir (str): Ruta de la carpeta outputs
        backup (bool): Si crear copias de respaldo antes de formatear
    """
    outputs_path = Path(outputs_dir)
    
    if not outputs_path.exists():
        print(f"Error: La carpeta {outputs_dir} no existe.")
        return
    
    # Buscar todos los archivos .txt
    txt_files = list(outputs_path.glob("*.txt"))
    
    if not txt_files:
        print("No se encontraron archivos .txt en la carpeta outputs.")
        return
    
    print(f"Encontrados {len(txt_files)} archivos .txt para formatear...")
    
    # Crear carpeta de respaldo si es necesario
    if backup:
        backup_dir = outputs_path / "backup_original"
        backup_dir.mkdir(exist_ok=True)
        print(f"Carpeta de respaldo: {backup_dir}")
    
    success_count = 0
    
    for txt_file in txt_files:
        # Crear respaldo si es necesario
        if backup:
            backup_path = backup_dir / txt_file.name
            try:
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as src:
                    with open(backup_path, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
            except Exception as e:
                print(f"⚠️  No se pudo crear respaldo de {txt_file.name}: {str(e)}")
        
        # Formatear el archivo
        if format_contract_file(str(txt_file)):
            success_count += 1
    
    print(f"\nResumen:")
    print(f"✓ Archivos formateados exitosamente: {success_count}")
    print(f"✗ Archivos con errores: {len(txt_files) - success_count}")
    
    if backup:
        print(f"📁 Copias de respaldo guardadas en: {backup_dir}")

def main():
    """Función principal del script."""
    # Obtener la ruta del directorio actual
    current_dir = Path(__file__).parent
    outputs_dir = current_dir / "outputs"
    
    print("=== Formateador de Archivos de Contratos ===")
    print(f"Directorio de outputs: {outputs_dir}")
    
    # Verificar si la carpeta outputs existe
    if not outputs_dir.exists():
        print(f"Error: No se encontró la carpeta 'outputs' en {current_dir}")
        return
    
    # Crear respaldos por defecto
    backup = True
    print("Creando copias de respaldo antes de formatear...")
    
    # Formatear todos los archivos
    format_all_outputs(str(outputs_dir), backup=backup)
    
    print("\n¡Formateo completado!")

def test_formatting():
    """Función para probar el formateo con un archivo de muestra."""
    current_dir = Path(__file__).parent
    test_file = current_dir / "outputs" / "contrato oriental wok.txt"
    
    if test_file.exists():
        print("Probando formateo con archivo de muestra...")
        result = format_contract_file(str(test_file), str(current_dir / "outputs" / "test_formatted.txt"))
        print(f"Resultado de la prueba: {'✓ Éxito' if result else '✗ Error'}")
    else:
        print("Archivo de prueba no encontrado")

if __name__ == "__main__":
    # Ejecutar prueba primero
    test_formatting()
    print("\n" + "="*50 + "\n")
    # Ejecutar formateo completo
    main()
