import numpy as np, pandas as pd, random
from pathlib import Path
from dtaidistance import dtw_ndim

FEATURES_DIR = Path(r"D:\VC\PIA\630DS\630LGH\Fitted\630_Features")
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
