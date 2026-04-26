# 🧬 DNA Model Studio – Diabetes Prediction

🚀 **Live Demo:**  
https://bioinfomatics-rkpqzj4ushwkkhfu28ruep.streamlit.app/

---

## 📌 Overview

A hybrid **Machine Learning + Deep Learning** system for predicting **Type 2 Diabetes** from DNA sequences.

The system combines:
- TF-IDF k-mer features (classical ML)
- Deep models (CNN, CNN-BiLSTM)
- Ensemble decision
- Explainability & reliability analysis

---

## 🧠 Features

- 🔍 Multi-model prediction (XGBoost, ExtraTrees, CNN, etc.)
- 🧠 Final ensemble decision (weighted risk)
- ⚠️ Reliability analysis (model disagreement)
- 🧬 Explainability:
  - k-mer importance
  - Saliency map for DNA regions

---

## 🧪 Method

- DNA → k-mer TF-IDF (k=4–6)
- Deep models learn sequence patterns
- **Similarity-aware split** to avoid data leakage
- Ensemble improves robustness

---

## 🚀 Run locally

```bash
pip install -r requirements.txt
streamlit run app.py