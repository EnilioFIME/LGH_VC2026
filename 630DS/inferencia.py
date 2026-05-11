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
words = data.get("words", classes)  # fallback a classes si no hay words

seq_arr = np.array(sequence, dtype=np.float64)
best_label, best_score = -1, -np.inf
for label, model in models.items():
    try:
        s = model.score(seq_arr)
        if not (np.isnan(s) or np.isinf(s)) and s > best_score:
            best_score, best_label = s, label
    except:
        pass

print(f"Palabra detectada: {words[best_label]}  (score: {best_score:.2f})"
      if best_label >= 0 else "No se pudo clasificar")
