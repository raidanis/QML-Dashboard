---
title: QML Dashboard Test
emoji: 📚
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
python_version: '3.12'
app_file: app.py
pinned: false
---

# QML Intrusion Detection & Zero-Day Lab

Interactive Streamlit dashboard to compare **quantum ML** (PennyLane) and classical detectors, including a **zero-day mode**
that holds one attack family out of training.

**Models:** QSVM-IQP · VQC (angle) · OC-QKernel (benign-only quantum one-class) · RF · SVM · MLP · OC-SVM  
**Dashboard:** model selector · held-out attack family · ROC / confusion matrix · zero-day detection rate · latency · live traffic replay · circuit diagrams

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data
The demo uses **synthetic** NSL-KDD-like data so it runs out of the box. For real results, download
[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) / [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html),
keep numeric features, add a `family` column (`benign`, `dos`, `probe`, `r2l`, `u2r`, ...), subsample (a few thousand rows), and upload the CSV in the sidebar
or save it as `data/sample/flows.csv`.

## Method
Features → StandardScaler → PCA (n_qubits) → scaled to [0, π]. Quantum kernels use state-vector fidelity |⟨φ(x)|φ(x′)⟩|².
Quantum circuits are kept at 3–6 qubits so inference stays fast on free hosting. Pre-train with `notebooks/01_train_and_evaluate.ipynb`.

## Deploy
1. Push to GitHub. 2. Create a Hugging Face Space (SDK: Streamlit) and add repo secrets `HF_TOKEN`, `HF_SPACE` (`user/space`); the workflow in `.github/workflows/sync.yml` syncs on push.
3. Embed in your portfolio: `<iframe src="https://USER-SPACE.hf.space" width="100%" height="900"></iframe>`.

## Roadmap
QRL adaptive-threshold agent · real NSL-KDD/CICIDS2017 benchmark tables · noise models.
