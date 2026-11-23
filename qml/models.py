import numpy as np, pennylane as qml
from pennylane import numpy as pnp
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def _states(X, kind):
    n = X.shape[1]
    @qml.qnode(qml.device("default.qubit", wires=n))
    def circ(x):
        if kind == "iqp": qml.IQPEmbedding(x, wires=range(n), n_repeats=2)
        else: qml.AngleEmbedding(x, wires=range(n))
        return qml.state()
    return np.array([circ(x) for x in X])

def _kernel(A, B): return np.abs(A.conj() @ B.T) ** 2   # fidelity kernel |<a|b>|^2

def draw_circuit(kind, n=4):
    x = np.linspace(0.1, 3, n)
    @qml.qnode(qml.device("default.qubit", wires=n))
    def circ(x):
        if kind == "iqp": qml.IQPEmbedding(x, wires=range(n), n_repeats=2)
        else:
            qml.AngleEmbedding(x, wires=range(n))
            qml.StronglyEntanglingLayers(np.zeros((2, n, 3)), wires=range(n))
        return qml.state()
    return qml.draw_mpl(circ)(x)

class QSVM:  # supervised quantum-kernel SVM (IQP embedding)
    thr = 0.0
    def fit(self, X, y):
        self.S = _states(X, "iqp")
        self.clf = SVC(kernel="precomputed", class_weight="balanced").fit(_kernel(self.S, self.S), y); return self
    def score(self, X): return self.clf.decision_function(_kernel(_states(X, "iqp"), self.S))

class OneClassQK:  # benign-only quantum-kernel one-class SVM (zero-day oriented)
    thr = 0.0
    def __init__(self, nu=0.05): self.nu = nu
    def fit(self, X, y=None):
        self.S = _states(X, "iqp")
        self.clf = OneClassSVM(kernel="precomputed", nu=self.nu).fit(_kernel(self.S, self.S)); return self
    def score(self, X): return -self.clf.decision_function(_kernel(_states(X, "iqp"), self.S))

class VQC:  # angle encoding + StronglyEntanglingLayers, trained with Adam
    thr = 0.5
    def __init__(self, layers=2, steps=60, bs=64, lr=0.05): self.L, self.steps, self.bs, self.lr = layers, steps, bs, lr
    def _qnode(self, n):
        @qml.qnode(qml.device("default.qubit", wires=n))
        def c(w, x):
            qml.AngleEmbedding(x, wires=range(n)); qml.StronglyEntanglingLayers(w, wires=range(n))
            return qml.expval(qml.PauliZ(0))
        return c
    def fit(self, X, y):
        n = X.shape[1]; c = self._qnode(n); rng = np.random.default_rng(0)
        w = pnp.array(rng.normal(0, .1, (self.L, n, 3)), requires_grad=True)
        opt = qml.AdamOptimizer(self.lr); y = np.asarray(y, float)
        pw = len(y) / (2 * max(y.sum(), 1)); nw = len(y) / (2 * max((1 - y).sum(), 1))
        for _ in range(self.steps):
            i = rng.choice(len(X), min(self.bs, len(X)), replace=False)
            Xb, yb = pnp.array(X[i], requires_grad=False), y[i]
            sw = np.where(yb == 1, pw, nw)
            w = opt.step(lambda w: pnp.mean(sw * (((1 - c(w, Xb)) / 2) - yb) ** 2), w)
        self.w = np.array(w); return self
    def score(self, X): return (1 - np.asarray(self._qnode(X.shape[1])(self.w, X))) / 2

class Sk:  # sklearn wrapper
    def __init__(self, est, thr=0.5, kind="proba"): self.est, self.thr, self.kind = est, thr, kind
    def fit(self, X, y=None):
        self.est.fit(X, y) if self.kind == "proba" else self.est.fit(X); return self
    def score(self, X):
        if self.kind == "proba": return self.est.predict_proba(X)[:, 1]
        return -self.est.decision_function(X)

REGISTRY = {
    "QSVM-IQP (quantum, supervised)":      {"mode": "supervised", "make": QSVM},
    "VQC-Angle (quantum, supervised)":     {"mode": "supervised", "make": VQC},
    "OC-QKernel (quantum, benign-only)":   {"mode": "one-class",  "make": OneClassQK},
    "Random Forest (classical)":           {"mode": "supervised", "make": lambda: Sk(RandomForestClassifier(200, class_weight="balanced", random_state=0))},
    "SVM-RBF (classical)":                 {"mode": "supervised", "make": lambda: Sk(SVC(probability=True, class_weight="balanced", random_state=0))},
    "MLP (classical)":                     {"mode": "supervised", "make": lambda: Sk(MLPClassifier((32, 16), max_iter=500, random_state=0))},
    "OC-SVM RBF (classical, benign-only)": {"mode": "one-class",  "make": lambda: Sk(OneClassSVM(nu=0.05, gamma="scale"), 0.0, "decision")},
}
