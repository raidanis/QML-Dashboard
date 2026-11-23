import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

FAMILIES = ["benign", "dos", "probe", "r2l", "u2r"]

def synthetic(n=2000, d=12, seed=0):
    """Synthetic NSL-KDD-like demo data. Replace with real samples (see README)."""
    centers = {f: np.random.default_rng(42 + i).normal(0, 0.8, d) for i, f in enumerate(FAMILIES)}
    centers["benign"] = np.zeros(d)
    rng = np.random.default_rng(seed)
    fam = rng.choice(FAMILIES, size=n, p=[.55, .2, .12, .08, .05])
    X = np.stack([centers[f] + rng.normal(0, 1, d) for f in fam])
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(d)])
    df["family"] = fam
    return df

def load_csv(path_or_buf):
    """CSV must be numeric features + a `family` column ('benign' or attack family name)."""
    df = pd.read_csv(path_or_buf)
    if "family" not in df.columns:
        raise ValueError("CSV needs a 'family' column (use 'benign' for normal traffic).")
    fam = df["family"].astype(str).str.lower()
    return df.select_dtypes("number").assign(family=fam)

class Prep:
    """Standardize -> PCA(n_qubits) -> scale to [0, pi] for quantum encoders."""
    def __init__(self, n_qubits=4): self.n = n_qubits
    def fit(self, X):
        self.sc = StandardScaler().fit(X)
        self.pca = PCA(self.n).fit(self.sc.transform(X))
        self.mm = MinMaxScaler((0, np.pi)).fit(self.pca.transform(self.sc.transform(X)))
        return self
    def transform(self, X):
        return np.clip(self.mm.transform(self.pca.transform(self.sc.transform(X))), 0, np.pi)
