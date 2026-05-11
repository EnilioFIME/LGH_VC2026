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
    input_dir = Path(r"D:\VC\PIA\630DS\WC630_Normalized")
    base_out = Path(r"D:\VC\PIA\630DS\630LGH\Irregular")
    
    dir_S = base_out / "630S"
    dir_Gx = base_out / "630_Gx"
    dir_Gy = base_out / "630_Gy"
    dir_Mag = base_out / "630_Mag"
    dir_Feat = base_out / "630_Features"
    
    w_width = 16
    cells_y, cells_x = 4, 4
    bins = 8
    angle_step = 2 * np.pi / bins
    bin_centers = np.linspace(0, 2 * np.pi, bins, endpoint=False)

    tiff_files = list(input_dir.glob("**/*.tiff"))
    print(f"Iniciando extracción LGH (Irregular) para {len(tiff_files)} imágenes...")

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
            
            #Gradiantes en direcciones x y
            gx = np.zeros_like(smoothed)
            gy = np.zeros_like(smoothed)
            gx[:, 1:-1] = smoothed[:, 2:] - smoothed[:, :-2]
            gy[1:-1, :] = smoothed[2:, :] - smoothed[:-2, :]
            
            #Magnitud y dirección angular
            mag = np.sqrt(gx**2 + gy**2)
            ang = np.arctan2(gy, gx)
            ang[ang < 0] += 2 * np.pi 
            
            # Guardado visual de pasos intermedios
            imwrite_unicode(dir_S / rel_folder / img_path.name, smoothed.astype(np.uint8))
            imwrite_unicode(dir_Gx / rel_folder / img_path.name, cv2.normalize(gx, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
            imwrite_unicode(dir_Gy / rel_folder / img_path.name, cv2.normalize(gy, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
            imwrite_unicode(dir_Mag / rel_folder / img_path.name, cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))

            # 4. Sliding Window & Irregular Grid (3 zonas: ascendentes/cuerpo/descendentes)
            word_features = []
            A, B, C = 1, 4, 1  # splits por zona
            
            for x_start in range(w - w_width + 1):
                x_end = x_start + w_width
                
                win_img = img[:, x_start:x_end]
                win_mag = mag[:, x_start:x_end]
                win_ang = ang[:, x_start:x_end]
                
                frame_vector = np.zeros((A + B + C) * cells_x * bins, dtype=np.float64)

                h_proj      = np.sum(win_img, axis=1).astype(np.float64)
                thresh_row  = 0.05 * np.max(h_proj) if np.max(h_proj) > 0 else 1
                active_rows = np.where(h_proj > thresh_row)[0]
                upperline   = int(active_rows[0])  if len(active_rows) > 0 else h // 4
                baseline    = int(active_rows[-1]) if len(active_rows) > 0 else 3 * h // 4

                zones  = [(0, upperline, A), (upperline, baseline, B), (baseline, h, C)]
                cell_w = w_width / cells_x
                vector_idx = 0

                for z_start, z_end, n_splits in zones:
                    z_h         = max(z_end - z_start, 1)
                    cell_h_zone = z_h / n_splits

                    for r in range(n_splits):
                        for c in range(cells_x):
                            r_start = z_start + int(r * cell_h_zone)
                            r_end   = z_start + int((r + 1) * cell_h_zone) if r < n_splits - 1 else z_end
                            c_start = int(c * cell_w)
                            c_end   = int((c + 1) * cell_w) if c < cells_x - 1 else w_width

                            cell_mag = win_mag[r_start:r_end, c_start:c_end].flatten()
                            cell_ang = win_ang[r_start:r_end, c_start:c_end].flatten()
                            hist = np.zeros(bins, dtype=np.float64)

                            for i in range(len(cell_mag)):
                                m, theta = cell_mag[i], cell_ang[i]
                                if m == 0:
                                    continue
                                distances = np.abs(theta - bin_centers)
                                distances = np.minimum(distances, 2 * np.pi - distances)
                                nearest   = np.argsort(distances)[:2]
                                alpha     = distances[nearest[0]]
                                hist[nearest[0]] += m * (1.0 - alpha / angle_step)
                                hist[nearest[1]] += m * (alpha / angle_step)

                            frame_vector[vector_idx:vector_idx + bins] = hist
                            vector_idx += bins

                eps = 1e-7
                sum_val = np.sum(frame_vector)
                frame_vector = frame_vector / sum_val if sum_val > eps else np.zeros_like(frame_vector)
                word_features.append(frame_vector)
            
            csv_path = dir_Feat / rel_folder / f"{img_path.stem}.csv"
            pd.DataFrame(word_features).to_csv(csv_path, header=False, index=False)
            
        except Exception as e:
            print(f"Error procesando {img_path.name}: {e}")

    print(f"\nLGH Irregular completada en {base_out}")

if __name__ == "__main__":
    process_lgh()
