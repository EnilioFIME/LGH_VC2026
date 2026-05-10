# Instrucciones del Proyecto: Word Spotting con LGH (RIMES 630)

> **Contexto:** Estamos implementando el paper *"Local Gradient Histogram Features for Word Spotting in Unconstrained Handwritten Documents"* (Rodríguez & Perronnin) usando 630 cartas del dataset RIMES. Te paso el archivo comprimido con las imágenes ya segmentadas por palabras. Tu trabajo cubre desde la limpieza de las imágenes hasta la experimentación con HMM y DTW.

---

## Resumen del pipeline completo

```
[ZIP de imágenes] → Paso 1: Extraer y revisar estructura
                  → Paso 2: Auditar y limpiar etiquetas
                  → Paso 3: Normalización (N630.py)
                  → Paso 4: LGH × 3 variantes de grilla (630LGH.py)
                  → Paso 5: Experimentación HMM
                  → Paso 6: Experimentación DTW
                  → Paso 7: Script de inferencia (input imagen → palabra)
```

---

## Paso 1 — Extraer y entender la estructura del ZIP

El archivo comprimido que te paso contiene recortes de palabras en formato `.tiff`, organizados en subcarpetas por carta/escritor. La estructura interna es la misma que genera `WC630.py`, que es algo así:

```
WC630/
├── 0001_L/
│   ├── 0001_L_0_0_monsieur.tiff
│   ├── 0001_L_0_3_contrat.tiff
│   └── ...
├── 0002_L/
│   └── ...
```

El nombre de cada archivo sigue el patrón: `{carta}_{renglon}_{posicion}_{palabra}.tiff`

**Qué hacer:**
1. Extrae el ZIP en una carpeta local, por ejemplo `C:\...\WC630`.
2. Abre algunas imágenes al azar para familiarizarte con la calidad de los recortes (algunas pueden estar mal segmentadas o contener ruido).
3. Identifica las 10 palabras que aparecen en los nombres de archivo (son las mismas top-10 que encontró `WC630.py`). Anótalas — las vas a usar como clases durante todo el experimento.

---

## Paso 2 — Auditar y limpiar las etiquetas (paso manual e importante)

Este paso es **crítico** y requiere revisión humana. El proceso automático de `WC630.py` extrajo las palabras basándose en posición en el XML, pero los recortes de imagen pueden no corresponder exactamente a la palabra etiquetada en el nombre del archivo.

### Qué revisar

Para cada clase de palabra (las 10), abre una muestra representativa de sus imágenes y verifica que:

- **El recorte contenga la palabra correcta.** Ejemplo: un archivo llamado `..._resiliation.tiff` debería mostrar visualmente la palabra "résiliation". Si ves otra palabra completamente diferente, el recorte es incorrecto.
- **El recorte no esté vacío ni sea basura** (solo ruido, líneas, manchas sin texto).
- **No haya recortes que contengan más de una palabra** (el segmentador a veces falla en palabras pegadas).

### Qué hacer con los casos problemáticos

| Caso | Acción |
|---|---|
| La imagen muestra claramente otra palabra dentro del top 10 | Corrige el archivo |
| La imagen muestra claramente otra palabra fuera del top 10 | Elimina el archivo |
| La imagen está vacía o es puro ruido | Elimina el archivo |
| La imagen está inclinada/mal recortada pero la palabra es legible | La normalizacion del Paso 3 probablemente la rescata — déjala |
| Duda razonable | En caso de duda, elimina; más vale tener menos datos limpios |

> **Tip práctico:** No es necesario revisar los miles de imágenes una por una. Revisa al menos 30-50 ejemplos por clase. Si una clase tiene muchos errores, revísala más a fondo.

### Llevar un registro

Crea un archivo `auditoria.txt` donde anotes cuántas imágenes eliminaste por clase y por qué razón (etiqueta incorrecta, vacía, etc.). Esto lo necesitaremos para el reporte.

---

## Paso 3 — Normalización con `N630.py`

Una vez limpio el dataset, aplica la normalización. El script `N630.py` ya está listo y hace lo siguiente en secuencia: binarización con Otsu, corrección de inclinación (skew), corrección de cursiva (slant), escalado al cuerpo principal del texto (target height = 18px) y eliminación de columnas vacías.

### Configuración de rutas

Abre `N630.py` y ajusta estas dos variables al inicio del `process_pipeline()`:

```python
base_path   = Path(r"C:\...\WC630")            # Donde extrajiste el ZIP
output_path = Path(r"C:\...\WC630_Normalized") # Donde se guardarán las imágenes normalizadas
```

### Ejecutar

```bash
python N630.py
```

El script recorre todas las subcarpetas recursivamente. Al terminar, `WC630_Normalized/` tendrá la misma estructura de subcarpetas con archivos renombrados `*_N.tiff`.

### Verificar resultados

Revisa visualmente algunas imágenes antes y después. Las normalizadas deben verse más "limpias", con altura uniforme de ~18px y sin columnas negras a los lados.

---

## Paso 4 — Extracción de características LGH (3 variantes de grilla)

El paper evalúa **tres tipos de grilla** para dividir la ventana deslizante. Tienes que generar las características LGH con cada una. El script base `630LGH.py` ya implementa la **Regular Fitted Grid** (la que da mejores resultados según el paper, mAP = 0.717 con HMM).

Necesitas crear **dos scripts adicionales** para las otras dos variantes.

### Ajuste global de tamaño de ventana

En los tres scripts, cambia el ancho de ventana de `w_width = 18` a **`w_width = 16`** (divisible entre 4, adecuado para grillas 4×4). Si prefieres más resolución, usa `w_width = 20`.

```python
w_width = 16  # antes era 18 — ahora es divisible entre 4
```

---

### Script A: `630LGH_fitted.py` — Regular Fitted Grid *(ya existe como `630LGH.py`)*

Este es el script base. Solo corrígele un **bug** importante antes de usarlo: la línea que normaliza usa `feature_vector` en lugar de `frame_vector`. Busca este bloque y corrígelo:

```python
# ❌ BUGGY (líneas actuales en 630LGH.py)
sum_val = np.sum(feature_vector)
if sum_val > eps:
    feature_vector /= sum_val
else:
    feature_vector = np.zeros_like(feature_vector)

# ✅ CORRECTO
sum_val = np.sum(frame_vector)
if sum_val > eps:
    frame_vector /= sum_val
else:
    frame_vector = np.zeros_like(frame_vector)
```

La grilla fitted ajusta dinámicamente la altura de la grilla al contenido real de la imagen (de `y_min` a `y_max`), lo que ya hace el script original.

Carpeta de salida sugerida: `630LGH/Fitted/`

---

### Script B: `630LGH_unfitted.py` — Regular Unfitted Grid

En este caso la grilla **no** se ajusta al contenido — usa toda la altura de la imagen (`0` a `h`). Copia `630LGH.py`, renómbralo `630LGH_unfitted.py`, cambia la carpeta de salida y reemplaza la sección de la grilla con esto:

```python
# En lugar de detectar y_min/y_max, usa siempre toda la altura
cell_h = h / cells_y          # usa h, no fit_h
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

        # ... el resto del cálculo del histograma es igual
```

Elimina también la condición `if len(y_indices) > 0:` que envuelve el cálculo — en unfitted siempre calculamos sobre toda la imagen.

Carpeta de salida sugerida: `630LGH/Unfitted/`

El paper reporta mAP = 0.321 para esta variante con la configuración `(1+4+1)×4`. Para reproducirlo exactamente usa `cells_y = 6` con las zonas de ascendentes/descendentes fijas, pero si quieres simplificarlo usa `cells_y = 4` y `cells_x = 4` directamente sobre la imagen completa.

---

### Script C: `630LGH_irregular.py` — Irregular Grid

Esta variante divide la ventana en **tres zonas verticales** separadas por la upperline y la baseline, y aplica divisiones independientes en cada zona. Es la más compleja de implementar.

Copia `630LGH.py`, renómbralo `630LGH_irregular.py` y reemplaza la sección de grilla con:

```python
# Parámetros de zonas (A filas arriba de upperline, B en cuerpo, C debajo de baseline)
A, B, C = 1, 4, 1   # según el paper: (A+B+C) × N celdas

# Detectar upperline y baseline mediante proyección horizontal
h_proj = np.sum(win_img, axis=1)
# Umbral simple: filas con más del 5% de píxeles activos
thresh_row = 0.05 * np.max(h_proj) if np.max(h_proj) > 0 else 1
active_rows = np.where(h_proj > thresh_row)[0]

if len(active_rows) > 0:
    upperline = active_rows[0]
    baseline  = active_rows[-1]
else:
    upperline = h // 4
    baseline  = 3 * h // 4

# Las tres zonas
zones = [
    (0,          upperline),   # zona ascendente (A divisiones)
    (upperline,  baseline),    # cuerpo principal (B divisiones)
    (baseline,   h),           # zona descendente (C divisiones)
]
zone_splits = [A, B, C]

vector_idx = 0
for zone_idx, (z_start, z_end) in enumerate(zones):
    n_splits = zone_splits[zone_idx]
    z_h = max(z_end - z_start, 1)
    cell_h_zone = z_h / n_splits
    cell_w = w_width / cells_x

    for r in range(n_splits):
        for c in range(cells_x):
            r_start = z_start + int(r * cell_h_zone)
            r_end   = z_start + int((r + 1) * cell_h_zone) if r < n_splits - 1 else z_end
            c_start = int(c * cell_w)
            c_end   = int((c + 1) * cell_w) if c < cells_x - 1 else w_width

            cell_mag = win_mag[r_start:r_end, c_start:c_end].flatten()
            cell_ang = win_ang[r_start:r_end, c_start:c_end].flatten()

            # ... cálculo del histograma igual que en el script base
            frame_vector[vector_idx : vector_idx + bins] = hist
            vector_idx += bins
```

El vector resultante tendrá `(A + B + C) * cells_x * bins = 6 * 4 * 8 = 192` dimensiones.

Carpeta de salida sugerida: `630LGH/Irregular/`

---

## Paso 5 — Experimentación con HMM

Crea un script `hmm_train_test.py`. El paper usa **5-fold cross-validation** con HMMs de izquierda a derecha.

### Librería recomendada

```bash
pip install hmmlearn scikit-learn numpy pandas
```

### Estructura del script

```python
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from hmmlearn import hmm
import pickle

# ── Configuración ──────────────────────────────────────────────
FEATURES_DIR = Path(r"C:\...\630LGH\Fitted\630_Features")
N_STATES     = 10       # 10 estados por palabra (igual que el paper)
N_FOLDS      = 5
N_ITER       = 100

# ── Carga de datos ─────────────────────────────────────────────
def load_sequences(word_class: str):
    """Carga todos los CSV de una clase como lista de arrays 2D."""
    sequences = []
    for csv_path in (FEATURES_DIR / word_class).glob("*.csv"):
        df = pd.read_csv(csv_path, header=None)
        if len(df) > 0:
            sequences.append(df.values.astype(np.float64))
    return sequences

# Descubre automáticamente las clases disponibles
classes = [d.name for d in FEATURES_DIR.iterdir() if d.is_dir()]
print(f"Clases encontradas: {classes}")

# Construye dataset completo: lista de (secuencia, etiqueta)
all_seqs, all_labels = [], []
for label, cls in enumerate(classes):
    seqs = load_sequences(cls)
    all_seqs.extend(seqs)
    all_labels.extend([label] * len(seqs))

all_labels = np.array(all_labels)

# ── 5-Fold Cross Validation ────────────────────────────────────
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(all_seqs)):
    print(f"\n── Fold {fold + 1}/{N_FOLDS} ──")

    # Separar train/test
    train_seqs   = [all_seqs[i] for i in train_idx]
    train_labels = all_labels[train_idx]
    test_seqs    = [all_seqs[i] for i in test_idx]
    test_labels  = all_labels[test_idx]

    # Entrenar un HMM por clase
    models = {}
    for label, cls in enumerate(classes):
        class_seqs = [s for s, l in zip(train_seqs, train_labels) if l == label]
        if not class_seqs:
            continue

        # Concatenar secuencias y registrar longitudes para hmmlearn
        X      = np.concatenate(class_seqs)
        lengths = [len(s) for s in class_seqs]

        model = hmm.GaussianHMM(
            n_components=N_STATES,
            covariance_type="diag",
            n_iter=N_ITER,
            init_params="stmc",
        )
        try:
            model.fit(X, lengths)
            models[label] = model
        except Exception as e:
            print(f"  Error entrenando {cls}: {e}")

    # Evaluar: el score es la log-verosimilitud del modelo ganador
    correct = 0
    for seq, true_label in zip(test_seqs, test_labels):
        best_label, best_score = -1, -np.inf
        for label, model in models.items():
            try:
                score = model.score(seq)
                if score > best_score:
                    best_score = score
                    best_label = label
            except Exception:
                pass
        if best_label == true_label:
            correct += 1

    acc = correct / len(test_seqs) if test_seqs else 0
    print(f"  Accuracy fold {fold + 1}: {acc:.4f}")
    fold_results.append(acc)

print(f"\n── Resultado final ──")
print(f"Accuracy promedio: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")

# Guardar modelos entrenados con todos los datos (para inferencia)
print("\nEntrenando modelos finales con todos los datos...")
final_models = {}
for label, cls in enumerate(classes):
    class_seqs = [s for s, l in zip(all_seqs, all_labels) if l == label]
    X      = np.concatenate(class_seqs)
    lengths = [len(s) for s in class_seqs]
    model = hmm.GaussianHMM(n_components=N_STATES, covariance_type="diag", n_iter=N_ITER)
    model.fit(X, lengths)
    final_models[label] = model

with open("hmm_models.pkl", "wb") as f:
    pickle.dump({"models": final_models, "classes": classes}, f)
print("Modelos guardados en hmm_models.pkl")
```

> **Nota:** El paper usa DET curves (False Acceptance vs False Rejection) como métrica, no accuracy pura. Para reproducirlo fielmente deberías calcular AP (Average Precision) por clase usando `sklearn.metrics.average_precision_score`. El código anterior simplifica esto a accuracy para que puedas tener resultados rápido — afina la métrica si el profesor lo requiere.

---

## Paso 6 — Experimentación con DTW

Crea un script `dtw_train_test.py`. El paper usa **5 imágenes aleatorias como queries** y repite el experimento **5 veces**.

### Librería recomendada

```bash
pip install dtaidistance scikit-learn numpy pandas
```

### Estructura del script

```python
import numpy as np
import pandas as pd
from pathlib import Path
from dtaidistance import dtw_ndim
import random

FEATURES_DIR = Path(r"C:\...\630LGH\Fitted\630_Features")
N_QUERIES    = 5
N_REPEATS    = 5

def load_sequences(word_class: str):
    sequences = []
    names = []
    for csv_path in (FEATURES_DIR / word_class).glob("*.csv"):
        df = pd.read_csv(csv_path, header=None)
        if len(df) > 0:
            sequences.append(df.values.astype(np.float64))
            names.append(csv_path.stem)
    return sequences, names

classes = [d.name for d in FEATURES_DIR.iterdir() if d.is_dir()]
class_data = {cls: load_sequences(cls) for cls in classes}

all_results = []

for repeat in range(N_REPEATS):
    print(f"\n── Repetición {repeat + 1}/{N_REPEATS} ──")

    # Seleccionar queries: 5 imágenes aleatorias por clase
    queries = {}
    for cls in classes:
        seqs, names = class_data[cls]
        if len(seqs) < N_QUERIES:
            continue
        indices = random.sample(range(len(seqs)), N_QUERIES)
        queries[cls] = [(seqs[i], names[i]) for i in indices]

    # Para cada imagen del dataset (que no sea query), calcular distancia a todas las queries
    correct, total = 0, 0
    for true_cls in classes:
        seqs, names = class_data[true_cls]
        for seq, name in zip(seqs, names):
            # Saltar si es query
            if true_cls in queries and name in [q[1] for q in queries[true_cls]]:
                continue

            best_cls, best_dist = None, np.inf
            for query_cls, query_list in queries.items():
                for q_seq, _ in query_list:
                    try:
                        dist = dtw_ndim.distance(seq, q_seq)
                        if dist < best_dist:
                            best_dist = dist
                            best_cls  = query_cls
                    except Exception:
                        pass

            if best_cls == true_cls:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0
    print(f"  Accuracy: {acc:.4f}")
    all_results.append(acc)

print(f"\n── Resultado final DTW ──")
print(f"Accuracy promedio: {np.mean(all_results):.4f} ± {np.std(all_results):.4f}")
```

---

## Paso 7 — Script de inferencia (input imagen → palabra predicha)

Una vez tienes los modelos HMM entrenados (`hmm_models.pkl`), crea `inferencia.py`:

```python
"""
Uso:
    python inferencia.py ruta/a/imagen.tiff
"""

import sys
import pickle
import numpy as np
import cv2
from pathlib import Path

# ── Importa las funciones de normalización y LGH ────────────────
# Asegúrate de que N630.py y 630LGH_fitted.py estén en el mismo directorio
# o agrégalos como módulos. Aquí se importan directamente.
from N630 import correct_skew, correct_slant, correct_size, remove_silence
# (Extrae la función de extracción de features del script LGH en una función separada)
from LGH_utils import extract_lgh_sequence  # ver nota abajo

def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

def normalize_image(img):
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    norm = correct_skew(binary)
    norm = correct_slant(norm)
    norm = correct_size(norm, target_height=18)
    norm = remove_silence(norm)
    return cv2.bitwise_not(norm)

def main():
    if len(sys.argv) < 2:
        print("Uso: python inferencia.py <ruta_imagen>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Error: no se encontró el archivo {img_path}")
        sys.exit(1)

    # 1. Leer y normalizar
    img = imread_unicode(img_path)
    if img is None:
        print("Error: no se pudo leer la imagen")
        sys.exit(1)
    normalized = normalize_image(img)

    # 2. Extraer características LGH
    sequence = extract_lgh_sequence(normalized)  # retorna array 2D (frames × features)
    if sequence is None or len(sequence) == 0:
        print("No se pudieron extraer características de la imagen")
        sys.exit(1)

    # 3. Cargar modelos y clasificar
    with open("hmm_models.pkl", "rb") as f:
        data = pickle.load(f)
    models  = data["models"]
    classes = data["classes"]

    seq_array = np.array(sequence, dtype=np.float64)
    best_label, best_score = -1, -np.inf
    for label, model in models.items():
        try:
            score = model.score(seq_array)
            if score > best_score:
                best_score = score
                best_label = label
        except Exception:
            pass

    if best_label >= 0:
        print(f"Palabra detectada: {classes[best_label]}  (score: {best_score:.2f})")
    else:
        print("No se pudo clasificar la imagen")

if __name__ == "__main__":
    main()
```

### Nota sobre `LGH_utils.py`

Para que `inferencia.py` funcione, necesitas refactorizar la lógica de extracción de features de `630LGH_fitted.py` en una función reutilizable. Crea `LGH_utils.py` con:

```python
def extract_lgh_sequence(img, w_width=16, cells_y=4, cells_x=4, bins=8):
    """
    Recibe una imagen normalizada (numpy array 2D, texto NEGRO sobre fondo BLANCO).
    Retorna la secuencia de vectores LGH como lista de arrays 1D.
    """
    import cv2, numpy as np

    # Invertir para que texto = blanco (255), fondo = negro (0)
    img_inv = cv2.bitwise_not(img)
    h, w = img_inv.shape
    if w < w_width:
        return []

    angle_step = 2 * np.pi / bins
    bin_centers = np.linspace(0, 2 * np.pi, bins, endpoint=False)

    smoothed = cv2.GaussianBlur(img_inv, (5, 5), 0).astype(np.float64)
    gx = np.zeros_like(smoothed)
    gy = np.zeros_like(smoothed)
    gx[:, 1:-1] = smoothed[:, 2:] - smoothed[:, :-2]
    gy[1:-1, :]  = smoothed[2:, :] - smoothed[:-2, :]
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)
    ang[ang < 0] += 2 * np.pi

    word_features = []
    for x_start in range(w - w_width + 1):
        x_end = x_start + w_width
        win_img = img_inv[:, x_start:x_end]
        win_mag = mag[:, x_start:x_end]
        win_ang = ang[:, x_start:x_end]

        y_indices = np.where(np.any(win_img > 0, axis=1))[0]
        frame_vector = np.zeros(cells_y * cells_x * bins)

        if len(y_indices) > 0:
            y_min, y_max = y_indices[0], y_indices[-1]
            fit_h  = y_max - y_min + 1
            cell_h = fit_h / cells_y
            cell_w = w_width / cells_x

            vector_idx = 0
            for r in range(cells_y):
                for c in range(cells_x):
                    r_start = y_min + int(r * cell_h)
                    r_end   = y_min + int((r+1)*cell_h) if r < cells_y-1 else y_max+1
                    c_start = int(c * cell_w)
                    c_end   = int((c+1)*cell_w) if c < cells_x-1 else w_width

                    cell_mag = win_mag[r_start:r_end, c_start:c_end].flatten()
                    cell_ang = win_ang[r_start:r_end, c_start:c_end].flatten()
                    hist = np.zeros(bins)

                    for i in range(len(cell_mag)):
                        m, theta = cell_mag[i], cell_ang[i]
                        if m == 0:
                            continue
                        distances = np.abs(theta - bin_centers)
                        distances = np.minimum(distances, 2*np.pi - distances)
                        nearest = np.argsort(distances)[:2]
                        alpha = distances[nearest[0]]
                        hist[nearest[0]] += m * (1.0 - alpha / angle_step)
                        hist[nearest[1]] += m * (alpha / angle_step)

                    frame_vector[vector_idx:vector_idx+bins] = hist
                    vector_idx += bins

        eps = 1e-7
        s = np.sum(frame_vector)
        frame_vector = frame_vector / s if s > eps else np.zeros_like(frame_vector)
        word_features.append(frame_vector)

    return word_features
```

---

## Resumen de archivos que debes crear/modificar

| Archivo | Acción |
|---|---|
| `630LGH.py` (existente) | Corregir bug `feature_vector` → `frame_vector`, cambiar `w_width=16` |
| `630LGH_unfitted.py` | Crear: grilla sin ajuste al contenido |
| `630LGH_irregular.py` | Crear: grilla con zonas upperline/baseline |
| `LGH_utils.py` | Crear: función `extract_lgh_sequence` reutilizable |
| `hmm_train_test.py` | Crear: entrenamiento y evaluación con HMM, 5-fold CV |
| `dtw_train_test.py` | Crear: evaluación con DTW, 5 queries × 5 repeticiones |
| `inferencia.py` | Crear: clasificación desde línea de comandos |
| `auditoria.txt` | Crear manualmente durante el Paso 2 |

---

## Orden de ejecución

```
1. Extraer ZIP                         (manual)
2. Revisar y limpiar imágenes          (manual + auditoria.txt)
3. python N630.py                      (normalización)
4. python 630LGH.py                    (features fitted)
5. python 630LGH_unfitted.py           (features unfitted)
6. python 630LGH_irregular.py          (features irregular)
7. python hmm_train_test.py            (guarda hmm_models.pkl)
8. python dtw_train_test.py            (resultados DTW)
9. python inferencia.py imagen.tiff    (prueba final)
```

---

## Notas finales

- Las **rutas hardcodeadas** en todos los scripts están en formato Windows (`C:\Users\emili\...`). Cámbialas a tu directorio local antes de ejecutar.
- Los modelos HMM pueden tardar varios minutos en entrenarse. Si el proceso es muy lento, reduce `N_ITER` a 50 como primera prueba.
- El paper reporta que el **fitted grid 4×4 con 8 bins** es la mejor configuración (mAP = 0.717 con HMM). Úsala como referencia para saber si tu implementación está en el rango correcto.
- Si necesitas comparar con los resultados exactos del paper, recuerda que ellos usan DET curves; la accuracy que calcula el script de arriba es una métrica más simple pero suficiente para verificar que todo funciona.
