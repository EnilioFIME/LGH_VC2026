# Instrucciones del Proyecto: Word Spotting con LGH (RIMES 630)

> **Contexto:** Implementamos el paper *"Local Gradient Histogram Features for Word Spotting"* (Rodríguez & Perronnin). Te paso un solo ZIP: `Images_Courriers.zip`. Tu trabajo es correr **todo** el pipeline desde ese archivo hasta tener los modelos entrenados y el script de inferencia funcionando.

---

## Estructura del proyecto (la vas a crear tú)

```
630DS/
├── Images_Courriers.zip       <- lo que yo te paso
├── R630L/                     <- se genera con R630L.py
├── WC630/                     <- se genera con WC630.py
├── WC630_Normalized/          <- se genera con N630.py
├── 630LGH/
│   ├── Fitted/                <- se genera con 630LGH_fitted.py
│   ├── Unfitted/              <- se genera con 630LGH_unfitted.py
│   └── Irregular/             <- se genera con 630LGH_irregular.py
└── (scripts .py aqui)
```

---

## Resumen del pipeline

```
Images_Courriers.zip
   -> R630L.py       : Extrae 630 cartas aleatorias (210 por DVD)
   -> WC630.py       : Lee XMLs, top-10 palabras, extrae recortes de imagen
   -> [MANUAL]       : Auditoría y limpieza de etiquetas incorrectas
   -> N630.py        : Normalización (binarización, skew, slant, tamaño, silence)
   -> 630LGH_*.py    : Features LGH en 3 variantes de grilla
   -> hmm/dtw scripts: Experimentación
   -> inferencia.py  : imagen -> palabra detectada
```

---

## Paso 1 — R630L.py: Muestreo de 630 cartas

Ajusta las rutas al inicio del script:

```python
zip_path   = Path(r"C:\...\630DS\Images_Courriers.zip")
output_dir = Path(r"C:\...\630DS\R630L")
```

Ejecuta: `python R630L.py`

Resultado: carpeta `R630L/` con subcarpetas `DVD1_TIF/`, `DVD2_TIF/`, `DVD3_TIF/` conteniendo 210 cartas cada una (imagen .tif + anotación .xml por carta).

> Agrega `random.seed(42)` antes del muestreo si necesitas reproducibilidad.

---

## Paso 2 — WC630.py: Conteo de palabras y extracción de recortes

Ajusta las rutas:

```python
base_path  = Path(r"C:\...\630DS\R630L\Images_Courriers")
zip_path   = Path(r"C:\...\630DS\imagettes_mots_cursif.zip")
output_dir = Path(r"C:\...\630DS\WC630")
```

Ejecuta: `python WC630.py`

El script imprime el Top 10 en consola y guarda los recortes en `WC630/` con esta estructura:

```
WC630/
├── 0001_L/
│   ├── 0001_L_0_0_monsieur.tiff
│   └── ...
```

Nombre de archivo: `{carta}_{renglon}_{posicion}_{palabra}.tiff`

**Anota el Top 10 que imprime el script — esas son tus 10 clases.**

---

## Paso 3 — Auditoría y limpieza (manual)

El segmentador de RIMES no es perfecto. Algunos recortes muestran la palabra equivocada, están vacíos o son ruido.

| Situación | Acción |
|-----------|--------|
| La imagen muestra claramente otra palabra | Eliminar |
| La imagen está vacía o es ruido | Eliminar |
| Imagen inclinada pero palabra legible | Dejar (la normalización lo rescata) |
| Duda razonable | Eliminar |

Revisa al menos 30-50 imágenes por clase. Crea `auditoria.txt` anotando cuántas eliminaste por clase y por qué.

---

## Paso 4 — N630.py: Normalización

Ajusta las rutas:

```python
base_path   = Path(r"C:\...\630DS\WC630")
output_path = Path(r"C:\...\630DS\WC630_Normalized")
```

Ejecuta: `python N630.py`

Genera `*_N.tiff` en `WC630_Normalized/` con la misma estructura de subcarpetas.

---

## Paso 5 — LGH: 3 variantes de grilla

**En los 3 scripts** cambia el ancho de ventana a un valor divisible entre 4:

```python
w_width = 16  # era 18
```

---

### A) 630LGH_fitted.py — Regular Fitted Grid

Copia `630LGH.py` -> `630LGH_fitted.py`.

**Corrige este bug primero** (variable equivocada en la normalización del frame):

```python
# BUGGY (como está actualmente)
sum_val = np.sum(feature_vector)
if sum_val > eps:
    feature_vector /= sum_val
else:
    feature_vector = np.zeros_like(feature_vector)

# CORRECTO
sum_val = np.sum(frame_vector)
if sum_val > eps:
    frame_vector /= sum_val
else:
    frame_vector = np.zeros_like(frame_vector)
```

Cambia las rutas:

```python
input_dir = Path(r"C:\...\630DS\WC630_Normalized")
base_out  = Path(r"C:\...\630DS\630LGH\Fitted")
```

La grilla ajusta su altura al contenido real de la ventana (y_min a y_max). Paper: **mAP = 0.717** con HMM.

---

### B) 630LGH_unfitted.py — Regular Unfitted Grid

Copia `630LGH_fitted.py` -> `630LGH_unfitted.py`. Cambia `base_out` a `630LGH/Unfitted/`.

Reemplaza el bloque `if len(y_indices) > 0:` y todo su contenido con:

```python
# Unfitted: usa siempre toda la altura de la imagen
cell_h = h / cells_y
cell_w = w_width / cells_x

vector_idx = 0
for r in range(cells_y):
    for c in range(cells_x):
        r_start = int(r * cell_h)
        r_end   = int((r + 1) * cell_h) if r < cells_y - 1 else h
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
```

Paper: **mAP = 0.321** con HMM.

---

### C) 630LGH_irregular.py — Irregular Grid

Copia `630LGH_fitted.py` -> `630LGH_irregular.py`. Cambia `base_out` a `630LGH/Irregular/`.

Divide la ventana en 3 zonas (ascendentes / cuerpo / descendentes) dando (1+4+1) x 4 = 24 celdas. Reemplaza el bloque de cálculo de grilla con:

```python
A, B, C = 1, 4, 1
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
```

Vector de 192 dimensiones. Paper: **mAP = 0.655** con HMM.

---

## Paso 6 — hmm_train_test.py

```
pip install hmmlearn scikit-learn numpy pandas
```

```python
import numpy as np, pandas as pd, pickle
from pathlib import Path
from sklearn.model_selection import KFold
from hmmlearn import hmm

# Cambia este path para evaluar cada variante (Fitted / Unfitted / Irregular)
FEATURES_DIR = Path(r"C:\...\630DS\630LGH\Fitted\630_Features")
N_STATES, N_FOLDS, N_ITER = 10, 5, 100

def load_sequences(word_class):
    seqs = []
    for csv_path in (FEATURES_DIR / word_class).glob("*.csv"):
        df = pd.read_csv(csv_path, header=None)
        if len(df) > 0:
            seqs.append(df.values.astype(np.float64))
    return seqs

classes = [d.name for d in FEATURES_DIR.iterdir() if d.is_dir()]
all_seqs, all_labels = [], []
for label, cls in enumerate(classes):
    for seq in load_sequences(cls):
        all_seqs.append(seq)
        all_labels.append(label)
all_labels = np.array(all_labels)

fold_accs = []
for fold, (train_idx, test_idx) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=42).split(all_seqs)):
    print(f"\n-- Fold {fold+1}/{N_FOLDS} --")
    train_seqs   = [all_seqs[i] for i in train_idx]
    train_labels = all_labels[train_idx]
    test_seqs    = [all_seqs[i] for i in test_idx]
    test_labels  = all_labels[test_idx]

    models = {}
    for label, cls in enumerate(classes):
        seqs = [s for s, l in zip(train_seqs, train_labels) if l == label]
        if not seqs:
            continue
        model = hmm.GaussianHMM(n_components=N_STATES, covariance_type="diag",
                                 n_iter=N_ITER, init_params="stmc")
        try:
            model.fit(np.concatenate(seqs), [len(s) for s in seqs])
            models[label] = model
        except Exception as e:
            print(f"  Error {cls}: {e}")

    correct = 0
    for seq, true_label in zip(test_seqs, test_labels):
        best_label, best_score = -1, -np.inf
        for label, model in models.items():
            try:
                s = model.score(seq)
                if s > best_score:
                    best_score, best_label = s, label
            except:
                pass
        if best_label == true_label:
            correct += 1

    acc = correct / len(test_seqs)
    print(f"  Accuracy: {acc:.4f}")
    fold_accs.append(acc)

print(f"\nAccuracy promedio HMM: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")

# Modelos finales con todos los datos (para inferencia)
final_models = {}
for label, cls in enumerate(classes):
    seqs = [s for s, l in zip(all_seqs, all_labels) if l == label]
    model = hmm.GaussianHMM(n_components=N_STATES, covariance_type="diag", n_iter=N_ITER)
    model.fit(np.concatenate(seqs), [len(s) for s in seqs])
    final_models[label] = model

with open("hmm_models.pkl", "wb") as f:
    pickle.dump({"models": final_models, "classes": classes}, f)
print("Guardado: hmm_models.pkl")
```

---

## Paso 7 — dtw_train_test.py

```
pip install dtaidistance numpy pandas
```

```python
import numpy as np, pandas as pd, random
from pathlib import Path
from dtaidistance import dtw_ndim

FEATURES_DIR = Path(r"C:\...\630DS\630LGH\Fitted\630_Features")
N_QUERIES, N_REPEATS = 5, 5

def load_sequences(word_class):
    seqs, names = [], []
    for csv_path in (FEATURES_DIR / word_class).glob("*.csv"):
        df = pd.read_csv(csv_path, header=None)
        if len(df) > 0:
            seqs.append(df.values.astype(np.float64))
            names.append(csv_path.stem)
    return seqs, names

classes    = [d.name for d in FEATURES_DIR.iterdir() if d.is_dir()]
class_data = {cls: load_sequences(cls) for cls in classes}
all_results = []

for repeat in range(N_REPEATS):
    print(f"\n-- Repeticion {repeat+1}/{N_REPEATS} --")
    queries = {}
    for cls in classes:
        seqs, names = class_data[cls]
        if len(seqs) >= N_QUERIES:
            idx = random.sample(range(len(seqs)), N_QUERIES)
            queries[cls] = [(seqs[i], names[i]) for i in idx]

    correct, total = 0, 0
    for true_cls in classes:
        seqs, names = class_data[true_cls]
        q_names = {n for _, n in queries.get(true_cls, [])}
        for seq, name in zip(seqs, names):
            if name in q_names:
                continue
            best_cls, best_dist = None, np.inf
            for q_cls, q_list in queries.items():
                for q_seq, _ in q_list:
                    try:
                        d = dtw_ndim.distance(seq, q_seq)
                        if d < best_dist:
                            best_dist, best_cls = d, q_cls
                    except:
                        pass
            if best_cls == true_cls:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0
    print(f"  Accuracy: {acc:.4f}")
    all_results.append(acc)

print(f"\nAccuracy promedio DTW: {np.mean(all_results):.4f} +/- {np.std(all_results):.4f}")
```

---

## Paso 8 — LGH_utils.py (función reutilizable para inferencia)

```python
import cv2, numpy as np

def extract_lgh_sequence(img, w_width=16, cells_y=4, cells_x=4, bins=8):
    """Imagen normalizada (texto negro / fondo blanco) -> lista de vectores LGH."""
    img_inv = cv2.bitwise_not(img)
    h, w    = img_inv.shape
    if w < w_width:
        return []

    angle_step  = 2 * np.pi / bins
    bin_centers = np.linspace(0, 2 * np.pi, bins, endpoint=False)
    smoothed    = cv2.GaussianBlur(img_inv, (5, 5), 0).astype(np.float64)
    gx = np.zeros_like(smoothed); gy = np.zeros_like(smoothed)
    gx[:, 1:-1] = smoothed[:, 2:] - smoothed[:, :-2]
    gy[1:-1, :]  = smoothed[2:, :] - smoothed[:-2, :]
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx); ang[ang < 0] += 2 * np.pi

    word_features = []
    for x_start in range(w - w_width + 1):
        win_img = img_inv[:, x_start:x_start + w_width]
        win_mag = mag[:,   x_start:x_start + w_width]
        win_ang = ang[:,   x_start:x_start + w_width]
        y_idx   = np.where(np.any(win_img > 0, axis=1))[0]
        fv      = np.zeros(cells_y * cells_x * bins)

        if len(y_idx) > 0:
            y_min, y_max = y_idx[0], y_idx[-1]
            cell_h = (y_max - y_min + 1) / cells_y
            cell_w = w_width / cells_x
            vi = 0
            for r in range(cells_y):
                for c in range(cells_x):
                    rs = y_min + int(r * cell_h)
                    re = y_min + int((r+1)*cell_h) if r < cells_y-1 else y_max+1
                    cs = int(c * cell_w)
                    ce = int((c+1)*cell_w) if c < cells_x-1 else w_width
                    cm = win_mag[rs:re, cs:ce].flatten()
                    ca = win_ang[rs:re, cs:ce].flatten()
                    hist = np.zeros(bins)
                    for i in range(len(cm)):
                        m, theta = cm[i], ca[i]
                        if m == 0:
                            continue
                        d = np.abs(theta - bin_centers)
                        d = np.minimum(d, 2*np.pi - d)
                        n2 = np.argsort(d)[:2]; alpha = d[n2[0]]
                        hist[n2[0]] += m * (1.0 - alpha / angle_step)
                        hist[n2[1]] += m * (alpha / angle_step)
                    fv[vi:vi+bins] = hist; vi += bins

        s = np.sum(fv)
        word_features.append(fv / s if s > 1e-7 else np.zeros_like(fv))
    return word_features
```

---

## Paso 9 — inferencia.py

Input por línea de comandos: `python inferencia.py ruta/imagen.tiff`

```python
import sys, pickle
import numpy as np, cv2
from pathlib import Path
from N630 import correct_skew, correct_slant, correct_size, remove_silence
from LGH_utils import extract_lgh_sequence

def imread_unicode(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

def normalize(img):
    _, b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n = correct_skew(b)
    n = correct_slant(n)
    n = correct_size(n, target_height=18)
    return cv2.bitwise_not(remove_silence(n))

if len(sys.argv) < 2:
    print("Uso: python inferencia.py <ruta_imagen>"); sys.exit(1)

img_path = Path(sys.argv[1])
if not img_path.exists():
    print(f"No se encontro: {img_path}"); sys.exit(1)

sequence = extract_lgh_sequence(normalize(imread_unicode(img_path)))
if not sequence:
    print("No se pudieron extraer caracteristicas"); sys.exit(1)

with open("hmm_models.pkl", "rb") as f:
    data = pickle.load(f)
models, classes = data["models"], data["classes"]

seq_arr = np.array(sequence, dtype=np.float64)
best_label, best_score = -1, -np.inf
for label, model in models.items():
    try:
        s = model.score(seq_arr)
        if s > best_score:
            best_score, best_label = s, label
    except:
        pass

print(f"Palabra detectada: {classes[best_label]}  (score: {best_score:.2f})"
      if best_label >= 0 else "No se pudo clasificar")
```

---

## Resumen de archivos

| Archivo | Accion |
|---------|--------|
| `R630L.py` | Ajustar rutas y ejecutar |
| `WC630.py` | Ajustar rutas y ejecutar |
| `N630.py` | Ajustar rutas y ejecutar |
| `630LGH_fitted.py` | Copiar `630LGH.py`, corregir bug `feature_vector->frame_vector`, `w_width=16` |
| `630LGH_unfitted.py` | Crear con grilla sin ajuste de altura |
| `630LGH_irregular.py` | Crear con grilla por zonas upperline/baseline |
| `LGH_utils.py` | Crear: funcion `extract_lgh_sequence` |
| `hmm_train_test.py` | Crear: 5-fold CV, genera `hmm_models.pkl` |
| `dtw_train_test.py` | Crear: 5 queries x 5 repeticiones |
| `inferencia.py` | Crear: clasificacion por linea de comandos |
| `auditoria.txt` | Llenar manualmente en el Paso 3 |

### Orden de ejecucion

```
python R630L.py
python WC630.py
[Auditoria manual de WC630/]
python N630.py
python 630LGH_fitted.py
python 630LGH_unfitted.py
python 630LGH_irregular.py
python hmm_train_test.py
python dtw_train_test.py
python inferencia.py imagen.tiff
```

---

## Notas

- Cambia todas las rutas `C:\...\` a tu directorio real antes de ejecutar.
- `random.seed(42)` en `R630L.py` para reproducibilidad del muestreo.
- Para pruebas rapidas de HMM usa `N_ITER = 50`.
- Para evaluar las 3 variantes, cambia `FEATURES_DIR` en los scripts de HMM y DTW apuntando a `Fitted`, `Unfitted` o `Irregular` segun corresponda.
