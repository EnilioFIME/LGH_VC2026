import numpy as np, pandas as pd, pickle, warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import GroupKFold
from hmmlearn import hmm

warnings.filterwarnings("ignore")          # quitar warnings ruidosos de hmmlearn

FEATURES_DIR = Path(r"D:\VC\PIA\630DS\630LGH\Fitted\630_Features")
XML_BASE_PATH = Path(r"D:\VC\PIA\630DS\R630L")
MAX_STATES   = 4        # maximo estados — con ~3 muestras/clase, mas es contraproducente
N_FOLDS      = 5
N_ITER       = 30
MIN_VAR      = 1e-3     # piso de varianza para evitar covarianzas cero

# ─── helpers ───────────────────────────────────────────────────────────────────

def get_word_from_filename(filename):
    """Extrae la palabra real del nombre de los CSVs.
    Patron: {docID}_L_{line}_{pos}_{WORD}_N.csv -> WORD"""
    parts = filename.stem.split("_")
    if len(parts) >= 3 and parts[-1] == "N":
        return parts[-2]
    return None

def build_writer_map():
    """Parsea todos los XMLs y construye un mapeo {docID: writer_name}."""
    writer_map = {}
    for xml_path in XML_BASE_PATH.glob("**/*.xml"):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            writer = root.find("writer")
            if writer is not None and writer.text:
                doc_id = xml_path.stem  # e.g. "00036_L"
                writer_map[doc_id] = writer.text.strip()
        except Exception:
            pass
    return writer_map

def choose_n_states(seqs, max_states=MAX_STATES):
    """Elige n_states seguro: al menos 2 frames por estado, minimo 2 estados."""
    total_frames = sum(len(s) for s in seqs)
    min_frames   = min(len(s) for s in seqs)
    # No puede haber mas estados que el minimo de frames en cualquier secuencia
    safe = min(max_states, min_frames, total_frames // max(len(seqs), 1))
    return max(2, safe)

def get_covars_diag(model):
    """Extrae los valores diagonales de covars_ (getter devuelve full matrices)."""
    c = model.covars_   # shape (n_components, n_dim, n_dim)
    if c.ndim == 3:
        return np.array([np.diag(c[i]) for i in range(c.shape[0])])
    return c  # ya es (n_components, n_dim)

def sanitize_model(model):
    """Repara NaN en TODOS los parametros del modelo para que sea usable."""
    nc = model.n_components
    nf = model.n_features

    # startprob_
    if np.any(np.isnan(model.startprob_)):
        model.startprob_ = np.zeros(nc)
        model.startprob_[0] = 1.0        # left-to-right: empieza en estado 0

    # transmat_
    if np.any(np.isnan(model.transmat_)):
        t = np.zeros((nc, nc))
        for i in range(nc - 1):
            t[i, i]   = 0.5
            t[i, i+1] = 0.5
        t[-1, -1] = 1.0
        model.transmat_ = t
    else:
        row_sums = model.transmat_.sum(axis=1)
        for i in range(nc):
            if row_sums[i] == 0 or np.isnan(row_sums[i]):
                model.transmat_[i] = 1.0 / nc

    # means_ — reemplazar NaN con la media global de los datos
    if np.any(np.isnan(model.means_)):
        valid_mask = ~np.any(np.isnan(model.means_), axis=1)
        if np.any(valid_mask):
            global_mean = model.means_[valid_mask].mean(axis=0)
        else:
            global_mean = np.zeros(nf)
        for i in range(nc):
            if np.any(np.isnan(model.means_[i])):
                model.means_[i] = global_mean

    # covars_ — extraer diagonal, reemplazar NaN/cero, reasignar
    diag_covars = get_covars_diag(model)
    bad = np.isnan(diag_covars) | (diag_covars <= 0)
    if np.any(bad):
        diag_covars[bad] = MIN_VAR
    model.covars_ = diag_covars   # setter espera (n_components, n_dim)

    return model

def init_left_to_right(n_states):
    """Genera startprob y transmat left-to-right para mejor convergencia."""
    startprob = np.zeros(n_states)
    startprob[0] = 1.0
    transmat = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        transmat[i, i]   = 0.7
        transmat[i, i+1] = 0.3
    transmat[-1, -1] = 1.0
    return startprob, transmat

def train_hmm(seqs, n_states, n_iter=N_ITER):
    """Entrena un GaussianHMM con topologia left-to-right e inicializacion robusta."""
    startprob, transmat = init_left_to_right(n_states)

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        init_params="mc",          # solo inicializar means y covars automaticamente
        params="stmc",             # entrenar todo
    )
    model.startprob_ = startprob
    model.transmat_  = transmat

    X = np.concatenate(seqs)
    lengths = [len(s) for s in seqs]
    model.fit(X, lengths)

    # Post-entrenamiento: aplicar piso de varianza (extraer diag, clamp, reasignar)
    diag_covars = get_covars_diag(model)
    diag_covars = np.maximum(diag_covars, MIN_VAR)
    model.covars_ = diag_covars

    return sanitize_model(model)

def model_is_healthy(model):
    """Verifica que el modelo no tenga NaN en ningun parametro critico."""
    diag_covars = get_covars_diag(model)
    return (not np.any(np.isnan(model.means_)) and
            not np.any(np.isnan(diag_covars)) and
            not np.any(np.isnan(model.startprob_)) and
            not np.any(np.isnan(model.transmat_)) and
            np.all(diag_covars > 0))

def safe_score(model, seq):
    """Score que devuelve -inf en lugar de nan."""
    try:
        s = model.score(seq)
        if np.isnan(s) or np.isinf(s):
            return -np.inf
        return s
    except:
        return -np.inf

# ─── construir mapeo docID → writer ───────────────────────────────────────────
writer_map = build_writer_map()
print(f"Writers parseados: {len(writer_map)} documentos, {len(set(writer_map.values()))} autores únicos")

# Recopilar todos los datos
all_seqs_temp = []
all_words_temp = []
all_groups_temp = []

print("Cargando datos desde CSVs...")
for csv_path in FEATURES_DIR.glob("**/*.csv"):
    word = get_word_from_filename(csv_path)
    if not word:
        continue
    
    doc_id = csv_path.parent.name
    writer = writer_map.get(doc_id, f"unknown_{doc_id}")
    
    df = pd.read_csv(csv_path, header=None)
    if len(df) > 0:
        all_seqs_temp.append(df.values.astype(np.float64))
        all_words_temp.append(word)
        all_groups_temp.append(writer)

# Construir diccionario de clases únicas (alfabético)
class_words = sorted(set(all_words_temp))
word_to_label = {w: i for i, w in enumerate(class_words)}
classes = class_words  # Para compatibilidad con el resto del script

print(f"Palabras unicas encontradas: {classes}")

all_seqs = all_seqs_temp
all_labels = np.array([word_to_label[w] for w in all_words_temp])
all_groups = np.array(all_groups_temp)

print(f"Clases: {len(classes)}, Secuencias totales: {len(all_seqs)}")

# Estadisticas
from collections import Counter
label_counts = Counter(all_labels)
print(f"  Min muestras/clase: {min(label_counts.values())}")
print(f"  Max muestras/clase: {max(label_counts.values())}")
print(f"  Promedio: {np.mean(list(label_counts.values())):.1f}")
writer_counts = Counter(all_groups)
print(f"  Writers con secuencias: {len(writer_counts)}")

# ─── cross-validation ─────────────────────────────────────────────────────────

fold_accs = []
for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=N_FOLDS).split(all_seqs, groups=all_groups)):
    print(f"\n-- Fold {fold+1}/{N_FOLDS} (Writer-Independent) --")
    train_writers = set(all_groups[train_idx])
    test_writers  = set(all_groups[test_idx])
    overlap = train_writers & test_writers
    print(f"  Writers train: {len(train_writers)}, test: {len(test_writers)}, overlap: {len(overlap)}")
    train_seqs   = [all_seqs[i] for i in train_idx]
    train_labels = all_labels[train_idx]
    test_seqs    = [all_seqs[i] for i in test_idx]
    test_labels  = all_labels[test_idx]

    models = {}
    trained, failed, skipped = 0, 0, 0
    for label, cls in enumerate(classes):
        seqs = [s for s, l in zip(train_seqs, train_labels) if l == label]
        if not seqs:
            skipped += 1
            continue
        n_states = choose_n_states(seqs)
        try:
            m = train_hmm(seqs, n_states)
            if model_is_healthy(m):
                models[label] = m
                trained += 1
            else:
                # Intentar con menos estados
                for ns in range(n_states - 1, 1, -1):
                    m = train_hmm(seqs, ns)
                    if model_is_healthy(m):
                        models[label] = m
                        trained += 1
                        break
                else:
                    failed += 1
        except Exception as e:
            failed += 1

    print(f"  Modelos: {trained} OK, {failed} fallidos, {skipped} sin datos")

    correct = 0
    scorable = 0
    for seq, true_label in zip(test_seqs, test_labels):
        best_label, best_score = -1, -np.inf
        for label, model in models.items():
            s = safe_score(model, seq)
            if s > best_score:
                best_score, best_label = s, label
        if best_label >= 0:
            scorable += 1
        if best_label == true_label:
            correct += 1

    acc = correct / len(test_seqs) if len(test_seqs) > 0 else 0
    print(f"  Scorable: {scorable}/{len(test_seqs)}")
    print(f"  Accuracy: {acc:.4f}")
    fold_accs.append(acc)

print(f"\nAccuracy promedio HMM: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")

# ─── entrenamiento final ──────────────────────────────────────────────────────

print("\n== Entrenamiento final con todos los datos ==")
final_models = {}
healthy_count = 0
for label, cls in enumerate(classes):
    seqs = [s for s, l in zip(all_seqs, all_labels) if l == label]
    if not seqs:
        continue
    n_states = choose_n_states(seqs)
    try:
        m = train_hmm(seqs, n_states)
        if not model_is_healthy(m):
            for ns in range(n_states - 1, 1, -1):
                m = train_hmm(seqs, ns)
                if model_is_healthy(m):
                    break
        final_models[label] = m
        if model_is_healthy(m):
            healthy_count += 1
    except Exception as e:
        print(f"  Error final {cls}: {e}")

print(f"\nModelos saludables: {healthy_count}/{len(final_models)}")

with open("hmm_models.pkl", "wb") as f:
    pickle.dump({"models": final_models, "classes": classes, "words": class_words}, f)
print(f"Guardado: hmm_models.pkl ({len(final_models)} modelos)")

# Quick test: probar que al menos un modelo puede dar score
if final_models and all_seqs:
    test_seq = all_seqs[0]
    scores_ok = sum(1 for m in final_models.values() if safe_score(m, test_seq) > -np.inf)
    print(f"Test rapido: {scores_ok}/{len(final_models)} modelos pueden evaluar una secuencia")
