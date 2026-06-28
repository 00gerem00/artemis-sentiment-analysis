# **ARTEMIS II - X (Twitter) Sentiment Analysis**

Four-class sentiment analysis of tweets about NASA's Artemis II mission, comparing five NLP models from recurrent neural networks to transformers, with an interactive Dash dashboard.

> **Dataunderdogs** - Data Visualization and Text Mining, Universita Cattolica del Sacro Cuore, Milan (IT)

---

## **Overview**

This project classifies public reaction to NASA's Artemis II mission, the April 2026 crewed lunar flyby, using tweets collected across five mission phases (departure, flyby, photo day, return, and a targeted conspiracy-hunt collection). Five NLP models are trained, evaluated, and compared: a BiLSTM with GloVe embeddings, ULMFiT (AWD-LSTM), and three fine-tuned transformers (DistilBERT, RoBERTa, DeBERTa-v3). All results are surfaced through an interactive Dash dashboard featuring per-class metrics, confusion matrices, probability distributions, model comparisons, and a Live Test panel for real-time inference with optional LLM-generated explanations.

---

## **Repository Structure**

```
ARTEMIS_Sentiment_Analysis/
├── dashboard/
│   ├── app.py                           # Dash application entry point
│   └── assets/
│       └── custom.css
├── data/
│   ├── processed/
│   │   └── artemis_master_dataset.csv   # Labelled master dataset
│   ├── raw/                             # Raw scraped CSVs (5 collection phases)
│   │   ├── conspiracyhunt.csv
│   │   ├── departure.csv
│   │   ├── flyby.csv
│   │   ├── photoday.csv
│   │   └── return.csv
│   └── splits/                          # Train / val / test splits
│       ├── train_split.csv
│       ├── val_split.csv
│       └── test_split.csv
├── models/
│   ├── bilstm/                          # BiLSTM .h5 weights (included in repo)
│   ├── embeddings/                      # GloVe Twitter embeddings (downloaded)
│   ├── transformers/                    # DistilBERT / RoBERTa / DeBERTa weights (downloaded)
│   ├── ulmfit/                          # ULMFiT classifier (downloaded)
│   ├── label_encoder.pkl
│   ├── preprocessing_config.json
│   └── tokenizer.pkl
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_BiLSTM_ULMFiT.ipynb
│   ├── 04_DistilBERT_RoBERTa_DeBERTa.ipynb
│   └── 05_model_comparison.ipynb
├── results/
│   ├── bilstm/                          # Per-configuration metrics, confusion matrices, probs
│   ├── grid_search/
│   ├── transformers/
│   ├── ulmfit/
│   └── latency_cpu.json
├── download_models.py                   # Fetches heavy weights from Google Drive
└── requirements.txt
```

---

## **Sentiment Classes**

| Class | Definition |
|---|---|
| **Conspiratorial** | Tweets promoting hoax theories, staged-mission claims, or distrust of official sources. |
| **Critical/Skeptical** | Tweets expressing doubt, concern, or reasoned opposition without conspiracy framing. |
| **Enthusiastic** | Tweets showing excitement, pride, or strong support for the mission. |
| **Neutral** | Informational or news-style tweets with no strong positive or negative stance. |

---

## **Models**

| Model | Description |
|---|---|
| **BiLSTM** | Bidirectional LSTM trained from scratch with 100-dimensional GloVe Twitter embeddings. |
| **ULMFiT** | AWD-LSTM language model (fastai) fine-tuned via gradual unfreezing and discriminative learning rates. |
| **DistilBERT** | Distilled BERT base (uncased), fine-tuned on the 4-class task; optimised for speed. |
| **RoBERTa** | Optimised BERT pretraining, fine-tuned for a strong predictive-quality/size balance. |
| **DeBERTa-v3** | Disentangled-attention transformer (small), the highest predictive-quality model in the analysis. |

---

## **Requirements**

**Python 3.11 or 3.12 is required.** Python 3.13 is not supported: pinned libraries including `tensorflow`, `fastai`, `transformers`, and `numpy` do not yet have 3.13-compatible wheels.

All Python dependencies are listed in `requirements.txt`.

---

## **Setup**

### **Minimal: read the notebooks and explore the dashboard (pre-computed results)**

```bash
git clone https://github.com/00gerem00/ARTEMIS_Sentiment_Analysis.git
cd ARTEMIS_Sentiment_Analysis

# Create a virtual environment with Python 3.11 or 3.12
python3.12 -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

This is sufficient to run all notebooks and view all pre-computed results in the dashboard.

### **Full: enable Live Test and live comparison classification**

After completing the minimal setup, download the heavy model weights from Google Drive (~2.5 GB total):

```bash
python download_models.py
```

This fetches the GloVe Twitter embeddings, the ULMFiT classifier, and the three transformer weight files (DistilBERT, RoBERTa, DeBERTa-v3). The BiLSTM `.h5` weights are already included in the repository and do not need to be downloaded.

---

## **Running the Notebooks**

The notebooks work in two environments without any code changes:

- **Google Colab** - the `!pip install` cells at the top of each notebook install all dependencies automatically on a fresh Colab runtime.
- **Local** - after `pip install -r requirements.txt`, those same cells will report "already satisfied" and skip silently; no manual editing is needed.

Notebooks 01-04 cover data preparation, EDA, and model training. Notebook 05 contains the final cross-model comparison and aggregated evaluation results.

---

## **Running the Dashboard**

```bash
python dashboard/app.py
```

Then open http://127.0.0.1:8050 in your browser.

The dashboard runs on Windows, macOS, and Linux. A minimal setup (no downloaded model weights) is enough for all pre-computed views. The Live Test and comparison classification panels require the full setup.

---

## **LLM-Based Explanations**

### **How it works**

The Live Test section of the dashboard offers an optional post-hoc explanation feature powered by an LLM. This feature exists only in the dashboard and has no equivalent in the notebooks.

After a tweet is classified, the dashboard can call the Groq API (`llama-3.1-8b-instant`) to generate a short natural-language explanation of each model's prediction. The goal is to help interpret why a model assigned a particular class, not just what it predicted.

The system uses a two-phase design:

1. **Reference judgment (Phase 1)**: The LLM reads the raw tweet and independently assigns one of the four sentiment classes, with no knowledge of any model's prediction. This reference class serves as a neutral baseline.
2. **Per-model comparison (Phase 2)**: For each selected model, the LLM compares that model's predicted class to the Phase-1 reference and delivers a verdict: correct (prediction matches reference), wrong (prediction differs and the reference is a better fit), or defensible (for genuinely ambiguous tweets where both labels are reasonable).

Both phases use few-shot prompting with examples that cover correct predictions, wrong predictions, ambiguous tweets, and minority-class cases (Conspiratorial, Critical/Skeptical) to improve coverage of hard cases. Temperature is set to 0.2 for consistent, low-variance answers. Responses are capped at 2-3 sentences and framed as a critical independent assessment rather than a defence of the model's output.

### **Limitations**

The LLM has no access to the models' internal weights, attention patterns, or output probabilities. Its explanation is an independent post-hoc opinion based solely on the tweet text and the predicted label. It may itself be inaccurate, particularly for short or highly ambiguous tweets.

### **API key**

You supply your own Groq API key at runtime using the password field in the dashboard. The key is used for the current browser session only and is never written to disk or logged anywhere.

---

## **Authors**

**Dataunderdogs**

- [Mirko Dervishi](https://github.com/Mirko-hubgit) - 5409240
- [Matteo Gerevini](https://github.com/00gerem00) - 5411210
- [Andrea Grulla](https://github.com/grullaandrea-png) - 5407125
- [Lorenzo Meroni](https://github.com/lorenzomeroni02) - 5410127

---

## **Course**

Developed for Data Visualization and Text Mining course at Universita Cattolica del Sacro Cuore, Milan (IT).
