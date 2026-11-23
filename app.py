import time, numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
from qml.data import synthetic, load_csv, FAMILIES
from qml.models import REGISTRY, draw_circuit
from qml.experiment import run

st.set_page_config("QML Intrusion Detection Lab", "🛡️", layout="wide")
st.title("🛡️ Quantum ML Intrusion Detection & Zero-Day Lab")

with st.sidebar:
    st.header("Setup")
    up = st.file_uploader("Upload CSV (numeric features + `family` column)", type="csv")
    df, key = (load_csv(up), None) if up else (synthetic(), "demo")
    st.caption("Using uploaded data" if up else "Using synthetic demo data (see README for NSL-KDD/CICIDS2017)")
    fams = sorted(set(df["family"]) - {"benign"})
    holdout = st.selectbox("Zero-day family (held out of training)", ["none"] + fams, index=min(len(fams), 4))
    names = st.multiselect("Models", list(REGISTRY), default=list(REGISTRY)[:3] + ["Random Forest (classical)"])
    nq = st.slider("Qubits (PCA components)", 3, 6, 4)
    go = st.button("Train / load & evaluate", type="primary")

if go and names:
    with st.spinner("Running models..."):
        st.session_state["res"] = run(df, holdout, names, n_qubits=nq, cache_key=key)
        st.session_state["hold"] = holdout
res = st.session_state.get("res")
if not res:
    st.info("Pick models and a held-out attack family, then press the button."); st.stop()

t1, t2, t3, t4 = st.tabs(["Metrics", "Curves", "Live replay", "Circuits"])
with t1:
    M = pd.DataFrame({n: r["metrics"] for n, r in res.items()}).T
    st.dataframe(M.style.format("{:.3f}").highlight_max(axis=0, color="#1b5e20"), use_container_width=True)
    st.caption("Zero-day detection = recall on the held-out family. One-class models train on benign traffic only.")
with t2:
    c1, c2 = st.columns(2)
    fig, ax = plt.subplots()
    for n, r in res.items():
        fpr, tpr, _ = roc_curve(r["y"], r["score"]); ax.plot(fpr, tpr, label=n.split(" (")[0])
    ax.plot([0, 1], [0, 1], "k--"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=7); c1.pyplot(fig)
    pick = c2.selectbox("Confusion matrix for", list(res))
    cm = confusion_matrix(res[pick]["y"], res[pick]["pred"], labels=[0, 1])
    fig2, ax2 = plt.subplots(); ax2.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm): ax2.text(j, i, v, ha="center", va="center")
    ax2.set_xticks([0, 1], ["benign", "attack"]); ax2.set_yticks([0, 1], ["benign", "attack"])
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("True"); c2.pyplot(fig2)
with t3:
    pick = st.selectbox("Detector", list(res), key="live")
    n, speed = st.slider("Flows to replay", 10, 60, 25), st.slider("Delay (s)", 0.0, 0.5, 0.1)
    if st.button("▶ Replay traffic"):
        r = res[pick]; idx = np.random.default_rng(1).choice(len(r["y"]), n, replace=False); rows = []; box = st.empty()
        for i in idx:
            alert = r["pred"][i] == 1
            rows.append({"flow": int(i), "true family": r["family"][i], "verdict": "🚨 ALERT" if alert else "✅ ok",
                         "score": round(float(r["score"][i]), 3), "zero-day": r["family"][i] == st.session_state["hold"]})
            box.dataframe(pd.DataFrame(rows[::-1]), use_container_width=True); time.sleep(speed)
with t4:
    kind = {"IQP embedding (QSVM / OC-QKernel)": "iqp", "Angle + StronglyEntangling (VQC)": "vqc"}
    k = st.radio("Circuit", list(kind), horizontal=True)
    fig3, _ = draw_circuit(kind[k], nq); st.pyplot(fig3)
