import time, zlib, joblib, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from .data import Prep
from .models import REGISTRY

CACHE = Path(__file__).resolve().parent.parent / "models" / "cache"

def run(df, holdout="u2r", names=None, n_qubits=4, n_train=300, n_test=500, seed=0, cache_key=None):
    """holdout = attack family unseen in training ('none' disables). One-class models see benign only."""
    names = names or list(REGISTRY)
    feats = [c for c in df.columns if c != "family"]
    tr, te = train_test_split(df, test_size=0.3, stratify=df["family"], random_state=seed)
    te = te.sample(min(n_test, len(te)), random_state=seed)
    Xte, yte = te[feats].values, (te["family"] != "benign").astype(int).values
    zd = (te["family"] == holdout).values
    out = {}
    for name in names:
        spec = REGISTRY[name]
        f = CACHE / f"{cache_key}_{holdout}_{n_qubits}_{zlib.crc32(name.encode())}.joblib" if cache_key else None
        if f and f.exists():
            prep, model = joblib.load(f)
        else:
            t = tr[tr["family"] != holdout] if spec["mode"] == "supervised" else tr[tr["family"] == "benign"]
            t = t.sample(min(n_train, len(t)), random_state=seed)
            prep = Prep(n_qubits).fit(t[feats].values)
            model = spec["make"]().fit(prep.transform(t[feats].values), (t["family"] != "benign").astype(int).values)
            if f: f.parent.mkdir(parents=True, exist_ok=True); joblib.dump((prep, model), f)
        t0 = time.perf_counter(); s = model.score(prep.transform(Xte)); lat = (time.perf_counter() - t0) / len(Xte) * 1e3
        pred = (s > model.thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
        out[name] = dict(score=s, pred=pred, y=yte, family=te["family"].values, metrics={
            "Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred, zero_division=0),
            "Recall": recall_score(yte, pred), "F1": f1_score(yte, pred),
            "AUC": roc_auc_score(yte, s), "FPR": fp / max(fp + tn, 1),
            "Zero-day detection": float(pred[zd].mean()) if zd.any() else float("nan"),
            "Latency (ms/flow)": lat})
    return out
