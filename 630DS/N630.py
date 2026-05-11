import cv2
import numpy as np
from pathlib import Path
import math

def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags)

def imwrite_unicode(path, img):
    ext = Path(path).suffix
    _, buffer = cv2.imencode(ext, img)
    buffer.tofile(str(path))

def get_1d_otsu_threshold(data):
    """Calcula el umbral de Otsu para un arreglo 1D (histograma de densidades)"""
    max_val = int(np.max(data))
    if max_val == 0: return 0
    
    hist, _ = np.histogram(data, bins=max_val+1, range=(0, max_val+1))
    total = len(data)
    sum_total = np.sum(np.arange(max_val+1) * hist)
    
    weight_b, sum_b, var_max, threshold = 0, 0, 0, 0
    
    for t in range(max_val+1):
        weight_b += hist[t]
        if weight_b == 0: continue
        weight_f = total - weight_b
        if weight_f == 0: break
        
        sum_b += t * hist[t]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = t
            
    return threshold

def correct_skew(img):
    """Corrige la inclinación maximizando la proyección horizontal (línea base)."""
    h, w = img.shape
    best_angle = 0
    max_var = -1
    
    # Barrido conservador de -15 a 15 grados
    for angle in range(-15, 16):
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST)
        
        # EL FIX: axis=1 hace la proyección horizontal (suma por filas)
        proj = np.sum(rotated, axis=1) 
        var = np.var(proj)
        
        if var > max_var:
            max_var = var
            best_angle = angle
            
    M_best = cv2.getRotationMatrix2D((w//2, h//2), best_angle, 1.0)
    return cv2.warpAffine(img, M_best, (w, h), flags=cv2.INTER_NEAREST)

def correct_slant(img):
    """Corrige la cursiva aplicando una cizalladura (shear) que maximice la proyección."""
    h, w = img.shape
    best_angle = 0
    max_S = -1
    
    # Barrido de -30 a 30 grados
    for angle in range(-30, 31, 2):
        rad = np.deg2rad(angle)
        shear_factor = np.tan(rad)
        
        new_w = w + int(h * abs(shear_factor))
        offset_x = -h * shear_factor if shear_factor < 0 else 0
        M = np.float32([[1, shear_factor, offset_x], [0, 1, 0]])
        
        sheared = cv2.warpAffine(img, M, (new_w, h), flags=cv2.INTER_NEAREST)
        proj = np.sum(sheared, axis=0)
        # S(alpha) sum of squared vertical density (Vinciarelli)
        S = np.sum(proj.astype(np.float64) ** 2) 
        
        if S > max_S:
            max_S = S
            best_angle = angle
            
    rad_best = np.deg2rad(best_angle)
    shear_best = np.tan(rad_best)
    new_w = w + int(h * abs(shear_best))
    offset_x = -h * shear_best if shear_best < 0 else 0
    M_best = np.float32([[1, shear_best, offset_x], [0, 1, 0]])
    
    return cv2.warpAffine(img, M_best, (new_w, h), flags=cv2.INTER_NEAREST)

def correct_size(img, target_height=18):
    """Encuentra la upperline y baseline usando Otsu y escala el cuerpo principal."""
    h_dens = np.sum(img, axis=1) / 255.0  # Proyección horizontal (conteos de píxeles)
    threshold = get_1d_otsu_threshold(h_dens)
    
    # Identificar regiones contiguas que superan el umbral
    above_thresh = (h_dens > threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(above_thresh[:, None])
    
    if num_labels <= 1:
        return img # No se encontró texto válido
        
    # Encontrar la región con más píxeles (ignorando el fondo que es label 0)
    max_pixels = -1
    core_label = -1
    for i in range(1, num_labels):
        pixels = np.sum(h_dens[labels.flatten() == i])
        if pixels > max_pixels:
            max_pixels = pixels
            core_label = i
            
    if core_label != -1:
        indices = np.where(labels.flatten() == core_label)[0]
        upperline = indices[0]
        baseline = indices[-1]
        
        main_body = baseline - upperline
        if main_body > 0:
            scale = target_height / main_body
            new_w = max(1, int(img.shape[1] * scale))
            new_h = max(1, int(img.shape[0] * scale))
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            
    return img

def remove_silence(img):
    """Elimina las columnas que son puramente fondo (negro)."""
    v_dens = np.sum(img, axis=0)
    # Conservar solo las columnas donde la suma de píxeles sea mayor a 0
    return img[:, v_dens > 0]

def process_pipeline():
    base_path = Path(r"D:\VC\PIA\630DS\WC630")
    output_path = Path(r"D:\VC\PIA\630DS\WC630_Normalized")
    
    # Obtener todas las imágenes TIFF en subcarpetas
    tiff_files = list(base_path.glob("**/*.tiff"))
    print(f"Iniciando normalización de {len(tiff_files)} imágenes...")
    
    for img_path in tiff_files:
        try:
            # 1. Leer imagen en escala de grises
            img = imread_unicode(img_path)
            if img is None: continue
            
            # 2. Binarización (Fondo blanco, Letra negra inicialmente)
            # Usamos THRESH_BINARY_INV + OTSU para que el texto sea BLANCO (255) y fondo NEGRO (0)
            # El texto en blanco es necesario para que las sumas matemáticas funcionen.
            _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 3. Normalizaciones secuenciales
            norm_img = correct_skew(binary_img)
            norm_img = correct_slant(norm_img)
            norm_img = correct_size(norm_img, target_height=18)
            norm_img = remove_silence(norm_img)
            
            # 4. Invertir de vuelta (Fondo blanco, texto negro) para guardar
            final_img = cv2.bitwise_not(norm_img)
            
            # 5. Crear estructura de salida y guardar
            relative_folder = img_path.parent.name
            target_folder = output_path / relative_folder
            target_folder.mkdir(parents=True, exist_ok=True)
            
            new_filename = f"{img_path.stem}_N.tiff"
            imwrite_unicode(target_folder / new_filename, final_img)
            
        except Exception as e:
            print(f"Error procesando {img_path.name}: {e}")

    print(f"\n¡Normalización completada! Archivos guardados en: {output_path}")

if __name__ == "__main__":
    process_pipeline()