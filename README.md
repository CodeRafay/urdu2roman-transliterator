

# Urdu → Roman Urdu Transliterator

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Urdu2Roman Transliterator** is an NLP project that converts Urdu script into Roman Urdu using deep learning.
It is built in **PyTorch** with a custom **BPE tokenizer** (implemented from scratch), and a **BiLSTM encoder + LSTM decoder** seq2seq model.

---

## 🚀 Features

* Urdu → Roman Urdu transliteration
* **Custom BPE tokenizer** (no external libraries used)
* **BiLSTM encoder** + **LSTM decoder** seq2seq architecture
* Training & evaluation pipeline in PyTorch
* Evaluation with **BLEU**, **Perplexity**, and **CER**
* Multiple experiments with varying hyperparameters
* Interactive **Streamlit app** with public deployment

---

## 📊 Dataset

We use the **Urdu Ghazals dataset (Rekhta corpus)**.

* Contains parallel Urdu text and corresponding Roman Urdu transliterations.
* Preprocessed to normalize Urdu script and tokenize with custom BPE.
* Splits: **50% train, 25% validation, 25% test**.

---

## 🧪 Experiments

We train multiple models by varying:

* Embedding dimension
* Hidden size
* Number of layers (encoder/decoder)
* Dropout rate
* Learning rate and batch size

At least **3 experiments** are reported with metrics and qualitative examples. Logs are stored in `/experiments/`.

---

## 📈 Evaluation Metrics

* **BLEU Score** – quality of generated transliterations
* **Perplexity** – model confidence in predictions
* **CER (Character Error Rate)** – similarity of prediction to ground truth
* **Qualitative Analysis** – common errors, spelling variations

---

## 🛠️ Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/your-username/urdu2roman-transliterator.git
cd urdu2roman-transliterator
pip install -r requirements.txt
```

---

## ⚙️ Training

Train the model with default hyperparameters:

```bash
python src/train.py --config configs/default.yaml
```

You can adjust parameters like embedding size, hidden size, and dropout via configs or CLI arguments.

---

## ▶️ Streamlit App

Run the local demo:

```bash
streamlit run streamlit_app/app.py
```

Public demo: **[Live App Link](https://your-deployment-link.com)**

---

## 📂 Repository Structure

```
urdu2roman-transliterator/
├─ data/                 # raw and processed dataset
├─ notebooks/            # exploration & preprocessing notebooks
├─ src/
│  ├─ preprocess.py      # normalization & rule-based transliterator
│  ├─ bpe.py             # BPE tokenizer (from scratch)
│  ├─ model.py           # PyTorch BiLSTM encoder + LSTM decoder
│  ├─ train.py           # training loop
│  ├─ evaluate.py        # evaluation scripts
├─ experiments/          # experiment configs and logs
├─ streamlit_app/        # Streamlit UI
├─ reports/              # evaluation reports and results
├─ requirements.txt
└─ README.md
```

---

## 📜 Results (Sample)

| Experiment | Emb Dim | Hidden Size | Val BLEU | Test BLEU | CER  |
| ---------- | ------- | ----------- | -------- | --------- | ---- |
| Exp A      | 128     | 256         | 0.41     | 0.38      | 0.22 |
| Exp B      | 256     | 512         | 0.48     | 0.45      | 0.18 |
| Exp C      | 256     | 256         | 0.43     | 0.40      | 0.20 |

---

## ✍️ Blog & Report

A detailed write-up of methodology, experiments, and results is available in the project blog:
👉 [Blog Post](https://your-blog-link.com)

Also shared as a LinkedIn report per assignment requirements.

---

## 📌 Future Work

* Improve Roman Urdu spelling normalization
* Add attention mechanism (Luong / Bahdanau)
* Extend to Urdu → English translation

---

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

