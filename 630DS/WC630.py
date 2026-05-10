import os
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import re
import zipfile

# Configuración de ruta raíz
base_path = Path(r"C:\Users\emili\Downloads\PIA VC\630DS\R630L\Images_Courriers")

# Estructura: { palabra: [ {archivo, renglon, numero}, ... ] }
data_dict = {}

def clean_word(word):
    return re.sub(r'[^\w\s]', '', word).lower()

# Buscamos en todas las subcarpetas de manera recursiva
xml_files = list(base_path.glob("**/*.xml"))

print(f"Analizando {len(xml_files)} archivos XML...")

for xml_path in xml_files:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for box in root.findall(".//box"):
            text_type = box.find("type")
            text_content = box.find("text")
            
            if text_type is not None and text_type.text == "Corps de texte" and text_content is not None and text_content.text:
                
                raw_text = text_content.text
                text_fixed = raw_text.replace('\\n', ' \n ')
                
                # Esto evita que los símbolos ¤ y {} ensucien el conteo
                text_processed = re.sub(r'¤\{(.*?)/.*?\}¤', r'\1', text_fixed)
                
                
                lines = text_processed.split('\n')
                
                for line_idx, line_text in enumerate(lines):
                    # Separamos por espacios para obtener las palabras del renglón actual
                    words_in_line = line_text.split()
                    
                    for word_idx, raw_word in enumerate(words_in_line):
                        word_key = clean_word(raw_word)
                        
                        # Filtro de longitud (5 o más letras)
                        if len(word_key) >= 6:
                            info = {
                                "palabra": raw_word,
                                "archivo": Path(xml_path.name).stem,
                                "renglon": line_idx,    # Índice 0
                                "numero": word_idx      # Índice 0, relativo al renglón
                            }
                            
                            if word_key not in data_dict:
                                data_dict[word_key] = []
                            data_dict[word_key].append(info)
                            
    except Exception as e:
        print(f"Error en {xml_path.name}: {e}")



# Contar frecuencia
word_counts = Counter({word: len(occurs) for word, occurs in data_dict.items()})
top_10 = word_counts.most_common(10)

print("\n" + "="*65)
print(f"{'PALABRA':<15} | {'FREQ':<5} | {'UBICACIÓN (Índices 0)'}")
print("="*65)

for word, count in top_10:
    first = data_dict[word][0]
    location = f"Archivo: {first['archivo']}, R: {first['renglon']}, P: {first['numero']}"
    print(f"{word.upper():<15} | {count:<5} | {location}")

zip_path = Path(r"C:\Users\emili\Downloads\PIA VC\630DS\imagettes_mots_cursif.zip")
output_dir = Path(r"C:\Users\emili\Downloads\PIA VC\630DS\WC630")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nIniciando extracción de imágenes para el Top 10...")

with zipfile.ZipFile(zip_path, 'r') as z:
    all_files = z.namelist() # Lista de rutas internas del ZIP (ej: 'lot_1/.../0001_L_0_0.tiff')

    for word_key, count in top_10:
        print(f"Buscando imágenes para la palabra: {word_key.upper()} ({count} repeticiones)")
        
        # Por cada vez que aparece la palabra en los XML
        for occurrence in data_dict[word_key]:
            archivo = occurrence['archivo']     # Ej: 0001_L
            renglon = occurrence['renglon']     # Ej: 0
            numero = occurrence['numero']       # Ej: 0
            
            # Construimos el nombre que RIMES usa para los recortes
            # Formato: {nombre_archivo}_{renglon}_{posicion}.tiff
            target_filename = f"{archivo}_{renglon}_{numero}.tiff"
            new_filename = f"{archivo}_{renglon}_{numero}_{word_key}.tiff"

            
            # Buscamos este archivo dentro de la lista inmensa del ZIP
            # Comparar solo el nombre final, ignorando la carpeta 'lot_X'
            match = next((f for f in all_files if f.endswith(target_filename)), None)
            
            if match:
                # Definimos la subcarpeta para este archivo específico (ej: WC630/0001_L/)
                letter_folder = output_dir / archivo
                letter_folder.mkdir(exist_ok=True)
                
                # Extraemos el archivo
                with z.open(match) as source, open(letter_folder / new_filename, "wb") as target:
                    target.write(source.read())
            else:
                # Opcional: print(f"No se encontró la imagen: {target_filename}")
                pass

print(f"\n¡Proceso terminado! Revisa la carpeta {output_dir}")
