import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import math

def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags)

def imwrite_unicode(path, img):
    ext = Path(path).suffix
    _, buffer = cv2.imencode(ext, img)
    buffer.tofile(str(path))

def process_lgh():
    input_dir = Path(r"C:\Users\emili\Downloads\PIA VC\630DS\WC630_Normalized")
    base_out = Path(r"C:\Users\emili\Downloads\PIA VC\630DS\630LGH")
    
    dir_S = base_out / "630S"
    dir_Gx = base_out / "630_Gx"
    dir_Gy = base_out / "630_Gy"
    dir_Mag = base_out / "630_Mag"
    dir_Feat = base_out / "630_Features"
    
    w_width = 18
    cells_y, cells_x = 4, 4
    bins = 8
    angle_step = 2 * np.pi / bins
    bin_centers = np.linspace(0, 2 * np.pi, bins, endpoint=False)

    tiff_files = list(input_dir.glob("**/*.tiff"))
    print(f"Iniciando extracción LGH para {len(tiff_files)} imágenes...")

    for img_path in tiff_files:
        try:
            img = imread_unicode(img_path)
            if img is None: continue
            
            h, w = img.shape
            if w < w_width:
                continue

            rel_folder = img_path.parent.name
            
            for d in [dir_S, dir_Gx, dir_Gy, dir_Mag, dir_Feat]:
                (d / rel_folder).mkdir(parents=True, exist_ok=True)

            #Suavizado Gaussiano
            smoothed = cv2.GaussianBlur(img, (5, 5), 0).astype(np.float64)
            
            #Gradiantes en direcciones x y para calcular el cambio de intensidad en las direcciones correspondientes
            gx = np.zeros_like(smoothed)
            gy = np.zeros_like(smoothed)
            gx[:, 1:-1] = smoothed[:, 2:] - smoothed[:, :-2]
            gy[1:-1, :] = smoothed[2:, :] - smoothed[:-2, :]
            
            #Magniud de fuerza de los bordes y dirreccion angular donde esta apuntando el cambio
            mag = np.sqrt(gx**2 + gy**2)
            ang = np.arctan2(gy, gx) #Direccion
            ang[ang < 0] += 2 * np.pi 
            
            # Guardado visual de pasos intermedios a 0-255
            imwrite_unicode(dir_S / rel_folder / img_path.name, smoothed.astype(np.uint8))
            imwrite_unicode(dir_Gx / rel_folder / img_path.name, cv2.normalize(gx, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
            imwrite_unicode(dir_Gy / rel_folder / img_path.name, cv2.normalize(gy, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
            imwrite_unicode(dir_Mag / rel_folder / img_path.name, cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))

            # 4. Sliding Window & Fitted Grid
            word_features = []
            
            for x_start in range(w - w_width + 1):
                x_end = x_start + w_width
                
                win_img = img[:, x_start:x_end]
                win_mag = mag[:, x_start:x_end]
                win_ang = ang[:, x_start:x_end]
                
                y_indices = np.where(np.any(win_img > 0, axis=1))[0] #Pixel mas alto en la window
                
                frame_vector = np.zeros(cells_y * cells_x * bins, dtype=np.float64)
                
                if len(y_indices) > 0:
                    y_min, y_max = y_indices[0], y_indices[-1] #Pixel mas bajo en la window
                    fit_h = y_max - y_min + 1
                    
                    cell_h = fit_h / cells_y
                    cell_w = w_width / cells_x
                    
                    vector_idx = 0
                    for r in range(cells_y):
                        for c in range(cells_x):
                            r_start = y_min + int(r * cell_h)
                            r_end = y_min + int((r + 1) * cell_h) if r < cells_y - 1 else y_max + 1
                            c_start = int(c * cell_w)
                            c_end = int((c + 1) * cell_w) if c < cells_x - 1 else w_width
                            
                            cell_mag = win_mag[r_start:r_end, c_start:c_end].flatten()
                            cell_ang = win_ang[r_start:r_end, c_start:c_end].flatten()
                            
                            hist = np.zeros(bins, dtype=np.float64)

                            #Evaluamos píxel por píxel en la celda
                            for i in range(len(cell_mag)):
                                m = cell_mag[i] 
                                theta = cell_ang[i] 
                                
                                #Si la magnitud es 0 no aporta nada al histograma
                                if m == 0:
                                    continue
                                    
                                distances = np.abs(theta - bin_centers)
                                distances = np.minimum(distances, 2 * np.pi - distances)
                                
                                nearest_indices = np.argsort(distances)[:2]
                                idx_a = nearest_indices[0] #El bin más cercano
                                idx_b = nearest_indices[1] #El segundo bin más cercano
                                
                                alpha = distances[idx_a]
                                beta = angle_step
                                
                                voto_a = m * (1.0 - (alpha / beta))
                                voto_b = m * (alpha / beta)
                                
                                hist[idx_a] += voto_a
                                hist[idx_b] += voto_b
                            
                            frame_vector[vector_idx : vector_idx + bins] = hist
                            vector_idx += bins
                            
                #Frame Normalization
                eps = 1e-7
                sum_val = np.sum(feature_vector)

                if sum_val > eps:
                    feature_vector /= sum_val

                else:
                    feature_vector = np.zeros_like(feature_vector)
                    
                word_features.append(frame_vector)
            
            csv_path = dir_Feat / rel_folder / f"{img_path.stem}.csv"
            pd.DataFrame(word_features).to_csv(csv_path, header=False, index=False)
            
        except Exception as e:
            print(f"Error procesando {img_path.name}: {e}")

    print(f"\nLGH completada en {base_out}")

if __name__ == "__main__":
    process_lgh()