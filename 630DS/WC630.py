import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import re

# ─────────── Configuración ───────────
base_path = Path(r"D:\VC\PIA\630DS\R630L")
output_dir = Path(r"D:\VC\PIA\630DS\WC630")
output_dir.mkdir(parents=True, exist_ok=True)

# ─────────── Utilidades ───────────

def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags)

def imwrite_unicode(path, img):
    ext = Path(path).suffix
    _, buffer = cv2.imencode(ext, img)
    buffer.tofile(str(path))

def clean_word(word):
    return re.sub(r'[^\w\s]', '', word).lower()

def segment_lines(binary_img, min_gap=5, min_line_height=8):
    """Segmenta líneas usando proyección horizontal.
    binary_img: imagen binarizada con texto en BLANCO (255) y fondo en NEGRO (0).
    """
    h_proj = np.sum(binary_img, axis=1)
    
    # Umbral: filas con tinta significativa
    thresh = 0.02 * np.max(h_proj) if np.max(h_proj) > 0 else 1
    active = h_proj > thresh
    
    lines = []
    in_line = False
    start = 0
    
    for i in range(len(active)):
        if active[i] and not in_line:
            start = i
            in_line = True
        elif not active[i] and in_line:
            if i - start >= min_line_height:
                lines.append((start, i))
            in_line = False
    
    # Última línea si termina al borde
    if in_line and len(active) - start >= min_line_height:
        lines.append((start, len(active)))
    
    # Fusionar líneas con separación muy pequeña (evita splits por trazos finos)
    merged = []
    for s, e in lines:
        if merged and s - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    
    return merged


def segment_words(line_img_binary, min_gap=4, min_word_width=5):
    """Segmenta palabras dentro de una línea usando proyección vertical.
    line_img_binary: imagen binarizada de UNA línea (texto blanco, fondo negro).
    """
    v_proj = np.sum(line_img_binary, axis=0)
    
    active = v_proj > 0
    
    segments = []
    in_word = False
    start = 0

    for i in range(len(active)):
        if active[i] and not in_word:
            start = i
            in_word = True
        elif not active[i] and in_word:
            segments.append((start, i))
            in_word = False

    if in_word:
        segments.append((start, len(active)))

    if not segments:
        return []

    # Calcular gaps entre segmentos consecutivos
    gaps = []
    for i in range(1, len(segments)):
        gap = segments[i][0] - segments[i-1][1]
        gaps.append(gap)

    if not gaps:
        return [(segments[0][0], segments[0][1])]

    # Usar la mediana de los gaps como umbral para separar palabras
    median_gap = np.median(gaps)
    word_gap_thresh = max(min_gap, median_gap * 1.8)

    # Agrupar segmentos de tinta en palabras
    words = []
    current_start = segments[0][0]
    current_end = segments[0][1]

    for i in range(1, len(segments)):
        gap = segments[i][0] - current_end
        if gap >= word_gap_thresh:
            if current_end - current_start >= min_word_width:
                words.append((current_start, current_end))
            current_start = segments[i][0]
        current_end = segments[i][1]

    if current_end - current_start >= min_word_width:
        words.append((current_start, current_end))

    return words


def segment_words_with_target(line_img_binary, expected_count, min_word_width=5):
    """Segmenta palabras ajustando el umbral de gap para obtener el conteo esperado.
    Prueba diferentes multiplicadores del median_gap hasta encontrar uno que dé
    el número correcto de palabras. Esto mejora la alineación con el XML.
    """
    v_proj = np.sum(line_img_binary, axis=0)
    active = v_proj > 0

    segments = []
    in_word = False
    start = 0

    for i in range(len(active)):
        if active[i] and not in_word:
            start = i
            in_word = True
        elif not active[i] and in_word:
            segments.append((start, i))
            in_word = False

    if in_word:
        segments.append((start, len(active)))

    if not segments:
        return []

    gaps = []
    for i in range(1, len(segments)):
        gap = segments[i][0] - segments[i-1][1]
        gaps.append(gap)

    if not gaps:
        return [(segments[0][0], segments[0][1])]

    # Probar diferentes multiplicadores para encontrar el conteo esperado
    sorted_gaps = sorted(gaps)
    best_words = None
    best_diff = float('inf')

    # Probamos usar cada gap como umbral (es decir, probar cortar en cada gap diferente)
    candidate_thresholds = sorted(set(gaps))
    # También probamos multiplicadores del median
    median_gap = np.median(gaps)
    for mult in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
        candidate_thresholds.append(median_gap * mult)

    for thresh in candidate_thresholds:
        words = []
        cs = segments[0][0]
        ce = segments[0][1]

        for i in range(1, len(segments)):
            gap = segments[i][0] - ce
            if gap >= thresh:
                if ce - cs >= min_word_width:
                    words.append((cs, ce))
                cs = segments[i][0]
            ce = segments[i][1]

        if ce - cs >= min_word_width:
            words.append((cs, ce))

        diff = abs(len(words) - expected_count)
        if diff < best_diff:
            best_diff = diff
            best_words = words

        if diff == 0:
            break

    return best_words if best_words else []


# ─────────── Paso 1: Parsear XMLs, construir estructura por box ───────────

# Para la validación necesitamos saber cuántas líneas y palabras por línea tiene cada box.
# Estructura por box: { (xml_path, box_idx): { "lines": [ [word1, word2, ...], ...], "coords": {...} } }

# Estructura de conteo global: { palabra_limpia: [ {info}, ... ] }
data_dict = {}

# También guardamos la estructura completa por box para validar conteos
# Clave: (str(xml_path), box_x1, box_y1) -> { "text_lines": [...], "coords": {...} }
box_structure = {}

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
                
                # Limpia símbolos ¤{...}¤ 
                text_processed = re.sub(r'¤\{(.*?)/.*?\}¤', r'\1', text_fixed)
                
                lines_text = text_processed.split('\n')
                
                box_x1 = int(box.attrib["top_left_x"])
                box_y1 = int(box.attrib["top_left_y"])
                box_x2 = int(box.attrib["bottom_right_x"])
                box_y2 = int(box.attrib["bottom_right_y"])
                
                # Guardar estructura del box completa (todas las palabras por línea)
                box_key = (str(xml_path), box_x1, box_y1)
                all_words_per_line = []
                for line_text in lines_text:
                    words_in_line = line_text.split()
                    all_words_per_line.append(words_in_line)
                
                box_structure[box_key] = {
                    "text_lines": all_words_per_line,
                    "expected_line_count": len([l for l in all_words_per_line if l]),  # líneas no vacías
                    "all_line_count": len(all_words_per_line),
                }
                
                for line_idx, line_text in enumerate(lines_text):
                    words_in_line = line_text.split()
                    
                    for word_idx, raw_word in enumerate(words_in_line):
                        word_key = clean_word(raw_word)
                        
                        # Filtro de longitud (6 o más letras)
                        if len(word_key) >= 6:
                            info = {
                                "palabra": raw_word,
                                "archivo": Path(xml_path.name).stem,
                                "renglon": line_idx,
                                "numero": word_idx,
                                "expected_words_in_line": len(words_in_line),
                                "box_x1": box_x1,
                                "box_y1": box_y1,
                                "box_x2": box_x2,
                                "box_y2": box_y2,
                                "xml_path": xml_path,
                                "box_key": box_key,
                            }
                            
                            if word_key not in data_dict:
                                data_dict[word_key] = []
                            data_dict[word_key].append(info)
                            
    except Exception as e:
        print(f"Error en {xml_path.name}: {e}")

# ─────────── Paso 2: Top 10 ───────────

word_counts = Counter({word: len(occurs) for word, occurs in data_dict.items()})
top_10 = word_counts.most_common(10)

print("\n" + "="*65)
print(f"{'PALABRA':<15} | {'FREQ':<5} | {'UBICACIÓN (Índices 0)'}")
print("="*65)

for word, count in top_10:
    first = data_dict[word][0]
    location = f"Archivo: {first['archivo']}, R: {first['renglon']}, P: {first['numero']}"
    print(f"{word.upper():<15} | {count:<5} | {location}")

print("="*65)

# ─────────── Paso 3: Segmentar y guardar recortes CON VALIDACIÓN ───────────

print(f"\nIniciando segmentación de palabras para el Top 10...")

# Cache de imágenes leídas
img_cache = {}
# Cache de segmentaciones ya hechas: (tif_path, box_key) -> { seg_lines, seg_words_per_line }
seg_cache = {}

total_saved = 0
total_missed = 0
total_skipped_mismatch = 0

for word_key, count in top_10:
    saved_this_word = 0
    missed_this_word = 0
    mismatch_this_word = 0
    
    for occurrence in data_dict[word_key]:
        archivo = occurrence['archivo']
        renglon = occurrence['renglon']
        numero = occurrence['numero']
        expected_words = occurrence['expected_words_in_line']
        box_x1 = occurrence['box_x1']
        box_y1 = occurrence['box_y1']
        box_x2 = occurrence['box_x2']
        box_y2 = occurrence['box_y2']
        xml_path = occurrence['xml_path']
        box_key = occurrence['box_key']
        
        # Buscar el .tif correspondiente al XML
        tif_path = xml_path.with_suffix('.tif')
        if not tif_path.exists():
            missed_this_word += 1
            continue
        
        # Leer imagen (con cache)
        cache_key = str(tif_path)
        if cache_key not in img_cache:
            img = imread_unicode(tif_path)
            if img is None:
                missed_this_word += 1
                continue
            img_cache[cache_key] = img
        
        img = img_cache[cache_key]
        h_img, w_img = img.shape
        
        # ── Cache de segmentación por box ──
        seg_key = (cache_key, box_key)
        
        if seg_key not in seg_cache:
            # Recortar el box "Corps de texte"
            y1 = max(0, box_y1)
            y2 = min(h_img, box_y2)
            x1 = max(0, box_x1)
            x2 = min(w_img, box_x2)
            box_crop = img[y1:y2, x1:x2]
            
            if box_crop.size == 0:
                seg_cache[seg_key] = None
            else:
                # Binarizar
                _, binary = cv2.threshold(box_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Obtener conteo esperado de líneas del XML
                bs = box_structure.get(box_key)
                expected_total_lines = bs["all_line_count"] if bs else 0
                
                # Segmentar líneas
                seg_lines = segment_lines(binary)
                
                # Segmentar palabras por línea, usando conteo esperado del XML
                seg_words = {}
                for li, (ly1, ly2) in enumerate(seg_lines):
                    line_bin = binary[ly1:ly2, :]
                    
                    # Si el conteo de líneas coincide, usamos el conteo esperado de palabras
                    if bs and len(seg_lines) == expected_total_lines:
                        text_lines = bs["text_lines"]
                        # Mapear li a la línea correspondiente en el XML
                        # (las líneas del XML pueden incluir vacías)
                        expected_w = len(text_lines[li]) if li < len(text_lines) else 0
                        if expected_w > 0:
                            words_seg = segment_words_with_target(line_bin, expected_w)
                        else:
                            words_seg = segment_words(line_bin)
                    else:
                        words_seg = segment_words(line_bin)
                    
                    seg_words[li] = words_seg
                
                seg_cache[seg_key] = {
                    "lines": seg_lines,
                    "words": seg_words,
                    "box_crop": box_crop,
                    "binary": binary,
                    "line_count_match": (len(seg_lines) == expected_total_lines),
                }
        
        cached = seg_cache.get(seg_key)
        if cached is None:
            missed_this_word += 1
            continue
        
        seg_lines = cached["lines"]
        seg_words = cached["words"]
        box_crop = cached["box_crop"]
        line_count_ok = cached["line_count_match"]
        
        # ── VALIDACIÓN 1: ¿El índice de renglón existe? ──
        if renglon >= len(seg_lines):
            mismatch_this_word += 1
            continue
        
        # ── VALIDACIÓN 2: ¿El conteo de líneas coincide? ──
        if not line_count_ok:
            mismatch_this_word += 1
            continue
        
        # ── VALIDACIÓN 3: ¿El conteo de palabras en esta línea coincide? ──
        words_in_this_line = seg_words.get(renglon, [])
        if len(words_in_this_line) != expected_words:
            mismatch_this_word += 1
            continue
        
        # ── VALIDACIÓN 4: ¿El índice de palabra existe? ──
        if numero >= len(words_in_this_line):
            mismatch_this_word += 1
            continue
        
        # ── Recortar ──
        line_y1, line_y2 = seg_lines[renglon]
        line_gray = box_crop[line_y1:line_y2, :]
        
        word_x1, word_x2 = words_in_this_line[numero]
        word_crop = line_gray[:, word_x1:word_x2]
        
        if word_crop.size == 0:
            missed_this_word += 1
            continue
        
        # Guardar el recorte
        letter_folder = output_dir / archivo
        letter_folder.mkdir(exist_ok=True)
        
        new_filename = f"{archivo}_{renglon}_{numero}_{word_key}.tiff"
        imwrite_unicode(letter_folder / new_filename, word_crop)
        saved_this_word += 1
    
    total_saved += saved_this_word
    total_missed += missed_this_word
    total_skipped_mismatch += mismatch_this_word
    print(f"  {word_key.upper():<15} -> Guardados: {saved_this_word}/{count}  (mismatch: {mismatch_this_word}, no encontrados: {missed_this_word})")

print(f"\n{'='*65}")
print(f"Total de recortes guardados:         {total_saved}")
print(f"Total descartados por mismatch:      {total_skipped_mismatch}")
print(f"Total no encontrados (sin .tif):     {total_missed}")
print(f"Carpeta de salida:                   {output_dir}")
print(f"¡Proceso terminado!")
