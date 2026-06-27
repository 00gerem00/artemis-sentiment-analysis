"""
ARTEMIS II - Sentiment Analysis Dashboard
dashboard/app.py

Run from repo root:   python dashboard/app.py
Run from dashboard/:  python app.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
import sys
import json
import pickle
import time
import base64
import random
import pathlib
import threading
from pathlib import Path
from io import BytesIO
from collections import Counter

# ── Data / ML ─────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import (
    classification_report as sk_clf_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)

# ── Plotly / Dash ─────────────────────────────────────────────────────────────
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table, no_update
import dash_bootstrap_components as dbc

# ── Optional heavy dependencies ───────────────────────────────────────────────
try:
    import emoji as _emoji_lib
    EMOJI_OK = True
except ImportError:
    EMOJI_OK = False

try:
    from wordcloud import WordCloud as _WC
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    WC_OK = True
except ImportError:
    WC_OK = False

try:
    import spacy as _spacy
    SPACY_OK = True
except ImportError:
    SPACY_OK = False

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")   # env fallback; runtime key preferred
GROQ_MODEL      = "llama-3.1-8b-instant"               # swap model name here if needed
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS  = 120
try:
    from groq import Groq as _Groq
    GROQ_PKG_OK = True   # package importable; actual key comes from the user at runtime
except ImportError:
    GROQ_PKG_OK = False
GROQ_OK = GROQ_PKG_OK   # kept so other code that checks GROQ_OK still works

# ── Paths ─────────────────────────────────────────────────────────────────────
DASHBOARD_DIR = Path(__file__).resolve().parent
ROOT = DASHBOARD_DIR.parent

DATA_DIR      = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR    = DATA_DIR / "splits"
MODELS_DIR    = ROOT / "models"
RESULTS_DIR   = ROOT / "results"
# No wordcloud directory; clouds are generated in memory, never saved to disk.

MASTER_CSV      = PROCESSED_DIR / "artemis_master_dataset.csv"
TEST_CSV        = SPLITS_DIR / "test_split.csv"
LABEL_ENC_PATH  = MODELS_DIR / "label_encoder.pkl"
TOKENIZER_PATH  = MODELS_DIR / "tokenizer.pkl"
PREP_CFG_PATH   = MODELS_DIR / "preprocessing_config.json"
LATENCY_PATH    = RESULTS_DIR / "latency_cpu.json"
NLP_CACHE_PATH  = RESULTS_DIR / "eda_nlp_cache.json"   # read-only; never written at runtime

BILSTM_H5       = MODELS_DIR / "bilstm" / "BiLSTM_sd0.2_rd0.2_u32_lr0.0005.h5"
ULMFIT_PKL      = MODELS_DIR / "ulmfit" / "ulmfit_classifier.pkl"
DISTILBERT_DIR  = MODELS_DIR / "transformers" / "distilbert"
ROBERTA_DIR     = MODELS_DIR / "transformers" / "roberta"
DEBERTA_DIR     = MODELS_DIR / "transformers" / "deberta"

PROBS_PATHS = {
    "BiLSTM":     RESULTS_DIR / "bilstm" / "probs_bilstm.npy",
    "ULMFiT":     RESULTS_DIR / "ulmfit" / "probs_ulmfit.npy",
    "DistilBERT": RESULTS_DIR / "transformers" / "probs_distilbert.npy",
    "RoBERTa":    RESULTS_DIR / "transformers" / "probs_roberta.npy",
    "DeBERTa-v3": RESULTS_DIR / "transformers" / "probs_deberta.npy",
}

MODEL_FILE_MAP = {
    "BiLSTM":     BILSTM_H5,
    "ULMFiT":     ULMFIT_PKL,
    "DistilBERT": DISTILBERT_DIR / "model.safetensors",
    "RoBERTa":    ROBERTA_DIR / "model.safetensors",
    "DeBERTa-v3": DEBERTA_DIR / "model.safetensors",
}

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES = ["Conspiratorial", "Critical/Skeptical", "Enthusiastic", "Neutral"]
MODEL_NAMES = ["BiLSTM", "ULMFiT", "DistilBERT", "RoBERTa", "DeBERTa-v3"]
NN_MODELS = ["BiLSTM", "ULMFiT"]
TR_MODELS = ["DistilBERT", "RoBERTa", "DeBERTa-v3"]
TR_DIRS   = {"DistilBERT": DISTILBERT_DIR, "RoBERTa": ROBERTA_DIR, "DeBERTa-v3": DEBERTA_DIR}

CLASS_COLORS = {
    "Conspiratorial":     "#ef4444",
    "Critical/Skeptical": "#f97316",
    "Enthusiastic":       "#22c55e",
    "Neutral":            "#3b82f6",
}
PLOTLY_TEMPLATE = "plotly_dark"
CARD_STYLE = {
    "background": "#0f1629",
    "border": "1px solid #1e3a5f",
    "borderRadius": "10px",
    "padding": "16px",
}

# ══════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING: exact replica from notebook 01_data_preparation
# ══════════════════════════════════════════════════════════════════════════════

def clean_tweet_master(text: str) -> str:
    if not isinstance(text, str):
        return ""
    try:
        text = text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    if EMOJI_OK:
        text = _emoji_lib.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"&amp;|&lt;|&gt;", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def final_formatting(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_tweet(text: str) -> str:
    return final_formatting(clean_tweet_master(text))


def _apply_contractions(text: str, prep_cfg: dict) -> str:
    apostrophe_variants = prep_cfg.get("apostrophe_variants", ["'"])
    for variant in apostrophe_variants[1:]:
        text = text.replace(variant, "'")
    for contraction, expansion in prep_cfg.get("contraction_map", {}).items():
        pattern = r"\b" + re.escape(contraction) + r"\b"
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_or_none(path, loader, *args, **kwargs):
    try:
        return loader(path, *args, **kwargs)
    except Exception:
        return None


print("[startup] Loading data…")

df_master = _load_or_none(MASTER_CSV, pd.read_csv, encoding="utf-8")
df_test   = _load_or_none(TEST_CSV,   pd.read_csv)

# EDA dataframe: drop the 1 row with NaN Sentiment_label → 6,623 rows,
# matching notebook 02 cell-8 (df.dropna(subset=['Sentiment_label']))
df_eda = (
    df_master.dropna(subset=["Sentiment_label"]).reset_index(drop=True)
    if df_master is not None else None
)

label_encoder = None
if LABEL_ENC_PATH.exists():
    try:
        with open(LABEL_ENC_PATH, "rb") as f:
            label_encoder = pickle.load(f)
    except Exception as e:
        print(f"  [warn] label_encoder load failed: {e}")

prep_cfg = {}
if PREP_CFG_PATH.exists():
    try:
        with open(PREP_CFG_PATH) as f:
            prep_cfg = json.load(f)
    except Exception:
        pass

latency_data = {}
if LATENCY_PATH.exists():
    try:
        with open(LATENCY_PATH) as f:
            latency_data = json.load(f)
    except Exception:
        pass

y_true = None
if df_test is not None and label_encoder is not None:
    try:
        y_true = label_encoder.transform(df_test["label"].values)
    except Exception:
        y_true = None

# ── Compute per-model metrics from saved probs ─────────────────────────────
print("[startup] Computing model metrics…")

model_metrics = {}
for name, path in PROBS_PATHS.items():
    if not path.exists() or y_true is None:
        continue
    try:
        probs = np.load(path)
        y_pred = probs.argmax(axis=1)
        if len(y_pred) != len(y_true):
            continue
        report = sk_clf_report(
            y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
        model_metrics[name] = {
            "probs": probs,
            "y_pred": y_pred,
            "report": report,
            "cm": cm,
            "macro_f1":   round(f1_score(y_true, y_pred, average="macro",  zero_division=0), 4),
            "macro_prec": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "macro_rec":  round(recall_score(y_true, y_pred, average="macro",  zero_division=0), 4),
            "conspir_f1":  round(f1_score(y_true, y_pred, average=None, zero_division=0)[0], 4),
            "critical_f1": round(f1_score(y_true, y_pred, average=None, zero_division=0)[1], 4),
        }
    except Exception as e:
        print(f"  [warn] {name}: {e}")

# ── Missing heavy-model detection ─────────────────────────────────────────────
_heavy_model_files = {
    "ULMFiT":     ULMFIT_PKL,
    "DistilBERT": DISTILBERT_DIR / "model.safetensors",
    "RoBERTa":    ROBERTA_DIR / "model.safetensors",
    "DeBERTa-v3": DEBERTA_DIR / "model.safetensors",
}
_missing_heavy = [n for n, p in _heavy_model_files.items() if not p.exists()]


def _download_notice() -> html.Div:
    if not _missing_heavy:
        return html.Div()
    return dbc.Alert(
        [
            html.B("Model weights not found: "),
            f"Missing: {', '.join(_missing_heavy)}. ",
            "Run ", html.Code("python download_models.py"),
            " from the repo root to download them.",
            " BiLSTM is available without downloading.",
        ],
        color="warning",
        style={"fontSize": "0.85rem"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# EDA FIGURES  (computed at startup from CSV, always fast)
# ══════════════════════════════════════════════════════════════════════════════

def _empty_fig(msg="Data not available"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(color="#94a3b8", size=14))
    fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="#0f1629",
                      plot_bgcolor="#0f1629", height=300)
    return fig


def make_label_dist_fig():
    if df_eda is None:
        return _empty_fig()
    counts = df_eda["Sentiment_label"].value_counts().reindex(CLASSES, fill_value=0)
    total  = counts.sum()
    pcts   = (counts / total * 100).round(1)
    colors = [CLASS_COLORS[c] for c in CLASSES]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "pie"}]],
        subplot_titles=["Count per Class", "Share (%)"],
    )
    fig.add_trace(
        go.Bar(
            x=CLASSES, y=counts.values,
            marker_color=colors,
            text=[f"{v}<br>({p}%)" for v, p in zip(counts.values, pcts.values)],
            textposition="outside",
            name="Count",
        ), row=1, col=1
    )
    fig.add_trace(
        go.Pie(
            labels=CLASSES, values=counts.values,
            marker_colors=colors,
            hole=0.4,
            textinfo="label+percent",
            showlegend=False,
        ), row=1, col=2
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor="#0f1629", plot_bgcolor="#0f1629",
        height=380, margin=dict(t=40, b=10, l=10, r=10),
        showlegend=False,
        title_text="Overall Sentiment Distribution  (N = {:,})".format(total),
    )
    return fig


def make_phase_fig():
    if df_eda is None:
        return _empty_fig()
    phases = {
        "departure": "1 · Departure",
        "flyby":     "2 · Flyby",
        "return":    "3 · Return",
    }
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=list(phases.values()),
        shared_yaxes=False,
    )
    for col, (phase, title) in enumerate(phases.items(), start=1):
        sub = df_eda[df_eda["source"] == phase]
        counts = sub["Sentiment_label"].value_counts().reindex(CLASSES, fill_value=0)
        total  = counts.sum()
        for cls, val in zip(CLASSES, counts.values):
            pct = (val / total * 100) if total else 0
            fig.add_trace(
                go.Bar(
                    x=[cls], y=[val],
                    name=cls, legendgroup=cls,
                    showlegend=(col == 1),
                    marker_color=CLASS_COLORS[cls],
                    text=[f"{val}<br>{pct:.1f}%"],
                    textposition="outside",
                ),
                row=1, col=col,
            )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor="#0f1629", plot_bgcolor="#0f1629",
        height=400, margin=dict(t=50, b=10, l=10, r=10),
        barmode="group", legend_title="Sentiment",
        title_text="Sentiment by Mission Phase (Departure · Flyby · Return)",
    )
    fig.update_xaxes(tickangle=15, tickfont_size=10)
    return fig


def make_length_fig():
    if df_eda is None:
        return _empty_fig()
    df = df_eda.copy()
    df["char_count"] = df["cleaned_text"].astype(str).apply(len)
    df["word_count"] = df["cleaned_text"].astype(str).apply(lambda x: len(x.split()))

    char_median = int(df["char_count"].median())
    word_median = int(df["word_count"].median())

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Text Length Distribution (Characters)",
            "Word Count Distribution",
        ],
    )
    fig.add_trace(
        go.Histogram(
            x=df["char_count"], nbinsx=50, name="Characters",
            marker_color="#22c55e", opacity=0.75, showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=df["word_count"], nbinsx=50, name="Words",
            marker_color="#3b82f6", opacity=0.75, showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_vline(
        x=char_median, line_dash="dash", line_color="#ffd700", line_width=2.5,
        annotation_text=f"median = {char_median} chars",
        annotation_position="top right",
        annotation_font=dict(color="#ffd700", size=12),
        row=1, col=1,
    )
    fig.add_vline(
        x=word_median, line_dash="dash", line_color="#ffd700", line_width=2.5,
        annotation_text=f"median = {word_median} words",
        annotation_position="top right",
        annotation_font=dict(color="#ffd700", size=12),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Character Count", row=1, col=1)
    fig.update_xaxes(title_text="Word Count", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor="#0f1629", plot_bgcolor="#0f1629",
        height=400, margin=dict(t=60, b=30, l=50, r=30),
        title_text="Text Length Distributions  (entire cleaned dataset, N = {:,})".format(len(df)),
    )
    return fig


def make_tfidf_fig():
    if df_eda is None:
        return _empty_fig()
    texts = df_eda["cleaned_text"].fillna("").astype(str).tolist()
    stop  = list(ENGLISH_STOP_WORDS)
    vec   = TfidfVectorizer(max_features=3000, stop_words=stop, ngram_range=(1, 1))
    X     = vec.fit_transform(texts)
    scores = X.sum(axis=0).A1
    words  = vec.get_feature_names_out()
    top20  = pd.DataFrame({"word": words, "score": scores}) \
               .nlargest(20, "score") \
               .sort_values("score")

    fig = go.Figure(go.Bar(
        y=top20["word"], x=top20["score"], orientation="h",
        marker=dict(color=top20["score"], colorscale="Plasma"),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor="#0f1629", plot_bgcolor="#0f1629",
        height=500, margin=dict(l=120, t=40, b=10, r=10),
        title_text="Top 20 Terms: Global TF-IDF",
        xaxis_title="Cumulative TF-IDF Score",
    )
    return fig


def make_bigram_fig():
    if df_eda is None:
        return _empty_fig()

    # Prefer spaCy-lemmatized docs (matches notebook 02 cell-52):
    #   TfidfVectorizer(max_features=2000, ngram_range=(2,2))  - no stop_words;
    #   stops already removed during lemmatization.
    if _lemma_docs_no_artemis and len(_lemma_docs_no_artemis) == len(df_eda):
        corpus = _lemma_docs_no_artemis
        vec = TfidfVectorizer(max_features=2000, ngram_range=(2, 2))
    else:
        # Fallback: raw cleaned_text with explicit stop-word list
        stop = list(ENGLISH_STOP_WORDS) + [
            "artemis", "nasa", "moon", "mission", "space",
            "artemis2", "artemisii", "artemis 2",
        ]
        vec = TfidfVectorizer(max_features=2000, ngram_range=(2, 2), stop_words=stop)
        corpus = df_eda["cleaned_text"].fillna("").astype(str).tolist()

    X    = vec.fit_transform(corpus)
    feat = vec.get_feature_names_out()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"Top Bigrams: {c}" for c in CLASSES],
        horizontal_spacing=0.15, vertical_spacing=0.2,
    )
    colorscales = ["Blues", "Greens", "Oranges", "Purples"]
    for i, cls in enumerate(CLASSES):
        r, c = divmod(i, 2)
        idx  = df_eda[df_eda["Sentiment_label"] == cls].index.tolist()
        idx  = [j for j in idx if j < X.shape[0]]
        if not idx:
            continue
        cat_scores = X[idx].sum(axis=0).A1
        top10 = (
            pd.DataFrame({"bigram": feat, "score": cat_scores})
            .nlargest(10, "score")
            .sort_values("score")
        )
        fig.add_trace(
            go.Bar(
                x=top10["score"], y=top10["bigram"], orientation="h",
                marker=dict(color=top10["score"], colorscale=colorscales[i]),
                name=cls, showlegend=False,
            ),
            row=r + 1, col=c + 1,
        )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor="#0f1629", plot_bgcolor="#0f1629",
        height=700, margin=dict(l=20, r=20, t=60, b=20),
        title_text="Top Bigrams per Sentiment Class (TF-IDF, excl. domain anchors)",
    )
    fig.update_yaxes(categoryorder="total ascending")
    return fig


_lemma_docs_no_artemis: list = []   # populated by _compute_nlp_cache(); used for bigrams (artemis excluded)
_lemma_docs:            list = []   # populated by _compute_nlp_cache(); lemmas WITH artemis (for global word cloud)

print("[startup] Computing EDA figures…")
FIG_LABEL_DIST = make_label_dist_fig()
FIG_PHASE      = make_phase_fig()
FIG_LENGTH     = make_length_fig()
FIG_TFIDF      = make_tfidf_fig()
# FIG_BIGRAM computed after NLP cache load so lemma docs are available

# ══════════════════════════════════════════════════════════════════════════════
# WORD CLOUDS: generated in memory on demand, never written to disk
# ══════════════════════════════════════════════════════════════════════════════

DOMAIN_STOPS = {
    "artemis", "nasa", "moon", "mission", "space", "artemisii", "artemis2",
    "artemis 2", "lunar", "crew",
}
# Notebook 02 cell-82 excludes exactly these 5 domain words from per-class clouds
_WC_DOMAIN_STOPS_NB = frozenset({"artemis", "nasa", "moon", "mission", "space"})

_WC_CACHE: dict[str, str] = {}   # session cache: label key → base64 data URI


def _wc_b64(corpus: str, colormap: str = "plasma",
            stopwords=None, max_words: int = 80) -> str | None:
    """Render a word cloud to a base64 PNG data URI without touching disk."""
    if not WC_OK or not corpus.strip():
        return None
    if stopwords is None:
        stopwords = set(ENGLISH_STOP_WORDS) | DOMAIN_STOPS
    wc = _WC(
        width=1000, height=500, background_color="#0f1629",
        colormap=colormap, max_words=max_words, stopwords=stopwords,
        collocations=False,
    )
    wc.generate(corpus)
    fig, ax = _plt.subplots(figsize=(12, 6), facecolor="#0f1629")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    _plt.tight_layout(pad=0)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0f1629")
    _plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


# ══════════════════════════════════════════════════════════════════════════════
# NLP CACHE (spaCy-based, optional): read from disk, computed in memory only
# ══════════════════════════════════════════════════════════════════════════════

nlp_cache = {}


def _load_nlp_cache() -> dict:
    if NLP_CACHE_PATH.exists():
        try:
            with open(NLP_CACHE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


_NLP_CUSTOM_MAP = {"artemis2": "artemis", "artemisii": "artemis", "amp": ""}
_NLP_ARTEMIS_STOP = {"artemis"}


def _compute_nlp_cache() -> dict:
    global _lemma_docs_no_artemis, _lemma_docs
    if not SPACY_OK or df_eda is None:
        return {}
    try:
        nlp = _spacy.load("en_core_web_sm")
    except OSError:
        try:
            _spacy.cli.download("en_core_web_sm")
            nlp = _spacy.load("en_core_web_sm")
        except Exception:
            return {}

    # Use df_eda (6,623 rows) to match notebook 02 cell-8
    texts = df_eda["cleaned_text"].fillna("").astype(str).tolist()
    print(f"  [nlp] Processing {len(texts)} docs with spaCy…")
    docs = list(nlp.pipe(texts, batch_size=128))

    pos_counter = Counter()
    ner_counter = Counter()
    entity_by_label: dict[str, list] = {}
    sent_lengths, sents_per_doc = [], []

    for doc in docs:
        sents_per_doc.append(len(list(doc.sents)))
        for sent in doc.sents:
            sent_lengths.append(len(sent))
        for token in doc:
            if not token.is_punct and not token.is_space and token.pos_ != "SYM":
                pos_counter[token.pos_] += 1
        for ent in doc.ents:
            ner_counter[ent.label_] += 1
            # Notebook 02 cell-65: only include entities with len > 1 after strip
            clean_text = ent.text.strip().title()
            if len(clean_text) > 1:
                entity_by_label.setdefault(ent.label_, []).append(clean_text)

    pos_readable = {
        "NOUN": "Nouns", "VERB": "Verbs", "PROPN": "Proper Nouns",
        "ADJ": "Adjectives", "ADV": "Adverbs", "ADP": "Adpositions",
        "PRON": "Pronouns", "AUX": "Auxiliary", "DET": "Determiners",
        "NUM": "Numbers", "PART": "Particles", "SCONJ": "Subord. Conj.",
        "CCONJ": "Coord. Conj.", "INTJ": "Interjections", "X": "Other",
    }
    pos_list = [
        {"tag": pos_readable.get(k, k), "count": v}
        for k, v in pos_counter.most_common(15)
    ]
    ner_list = [{"label": k, "count": v} for k, v in ner_counter.most_common(15)]

    top_entities = {}
    for etype in ["ORG", "PERSON", "CARDINAL"]:
        top = Counter(entity_by_label.get(etype, [])).most_common(10)
        top_entities[etype] = [{"entity": e, "count": c} for e, c in top]

    # Build both lemma-doc variants in one pass; matches notebook 02 cell-42
    # _lemma_docs:            WITH 'artemis'  (global word cloud, notebook cell-78)
    # _lemma_docs_no_artemis: WITHOUT 'artemis' (bigrams, notebook cell-52)
    lems_with, lems_without = [], []
    for doc in docs:
        with_a: list[str] = []
        without_a: list[str] = []
        for token in doc:
            if not token.is_punct and not token.is_space and not token.is_stop:
                lemma = token.lemma_.lower().strip()
                lemma = _NLP_CUSTOM_MAP.get(lemma, lemma)
                if not lemma or len(lemma) <= 1 or lemma == "-pron-":
                    continue
                with_a.append(lemma)
                if lemma not in _NLP_ARTEMIS_STOP:
                    without_a.append(lemma)
        lems_with.append(" ".join(with_a))
        lems_without.append(" ".join(without_a))
    _lemma_docs            = lems_with
    _lemma_docs_no_artemis = lems_without

    return {
        "pos": pos_list,
        "ner": ner_list,
        "top_entities": top_entities,
        "sent_length_mean":  float(np.mean(sent_lengths)) if sent_lengths else 0,
        "sents_per_doc_mean": float(np.mean(sents_per_doc)) if sents_per_doc else 0,
        "total_sentences": len(sent_lengths),
    }


print("[startup] Loading NLP cache…")
nlp_cache = _load_nlp_cache()
# spaCy NER processing is deferred: computed on first visit to the Linguistics & NER tab
_nlp_spacy_computed = False   # True once _compute_nlp_cache() has run in this session
_nlp_lock           = threading.Lock()

FIG_BIGRAM = make_bigram_fig()   # fallback (cleaned_text) until NER tab is first visited


def make_pos_fig():
    data = nlp_cache.get("pos", [])
    if not data:
        return _empty_fig("spaCy NLP cache not available.\nRun the app with spaCy installed to compute.")
    df = pd.DataFrame(data).sort_values("count")
    fig = go.Figure(go.Bar(
        x=df["count"], y=df["tag"], orientation="h",
        marker=dict(color=df["count"], colorscale="Viridis"),
        text=df["count"], textposition="outside",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor="#0f1629", plot_bgcolor="#0f1629",
        height=450, margin=dict(l=20, r=20, t=40, b=10),
        title_text="Part-of-Speech Distribution (Top 15)",
        xaxis_title="Token Count",
    )
    return fig


def make_ner_type_fig():
    data = nlp_cache.get("ner", [])
    if not data:
        return _empty_fig("spaCy NLP cache not available.")
    df = pd.DataFrame(data).sort_values("count")
    fig = go.Figure(go.Bar(
        x=df["count"], y=df["label"], orientation="h",
        marker=dict(color=df["count"], colorscale="Magma"),
        text=df["count"], textposition="outside",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor="#0f1629", plot_bgcolor="#0f1629",
        height=430, margin=dict(l=20, r=20, t=40, b=10),
        title_text="Named Entity Type Distribution",
        xaxis_title="Frequency",
    )
    return fig


def make_top_entities_fig():
    top = nlp_cache.get("top_entities", {})
    if not top:
        return _empty_fig("spaCy NLP cache not available.")
    etypes = ["ORG", "PERSON", "CARDINAL"]
    titles = ["Top Organizations (ORG)", "Top Persons (PERSON)", "Top Cardinals (CARDINAL)"]
    colors = ["Blues", "Purples", "Greens"]

    fig = make_subplots(rows=1, cols=3, subplot_titles=titles, horizontal_spacing=0.12)
    for col, (etype, cmap) in enumerate(zip(etypes, colors), start=1):
        data = top.get(etype, [])
        if not data:
            continue
        df = pd.DataFrame(data).sort_values("count")
        fig.add_trace(
            go.Bar(
                x=df["count"], y=df["entity"], orientation="h",
                marker=dict(color=df["count"], colorscale=cmap),
                showlegend=False,
            ),
            row=1, col=col,
        )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor="#0f1629", plot_bgcolor="#0f1629",
        height=420, margin=dict(l=20, r=20, t=50, b=10),
        title_text="Top Named Entities by Type",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# LAZY MODEL LOADING & INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

_model_cache: dict = {}
_model_lock  = threading.Lock()


def _model_file_ok(name: str) -> bool:
    path = MODEL_FILE_MAP.get(name)
    return path is not None and path.exists()


def _bilstm_h5py_fallback():
    """Rebuild BiLSTM architecture and load weights from .h5 via h5py.

    Used when tensorflow.keras load_model() fails with a config-deserialization
    error (e.g. 'quantization_config' unknown in Keras 3.x).  Locates weight
    datasets dynamically so it does not rely on save-specific layer-suffix names.
    """
    import h5py  # type: ignore
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.layers import (  # type: ignore
        Embedding, SpatialDropout1D, Bidirectional, LSTM, Dense,
    )

    max_len = prep_cfg.get("max_len", 60)

    # Walk the h5 file and collect every leaf dataset.
    all_datasets: dict[str, np.ndarray] = {}
    with h5py.File(str(BILSTM_H5), "r") as hf:
        root = hf.get("model_weights", hf)  # fall back to root if key absent

        def _collect(name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset):
                all_datasets[name] = obj[:]

        root.visititems(_collect)

    emb_w = None
    fw_k = fw_rk = fw_b = None
    bw_k = bw_rk = bw_b = None
    d_k = d_b = None

    for path, data in all_datasets.items():
        lo = path.lower()
        base = path.split("/")[-1]
        if ":" in base:  # strip TF tensor-index suffix (e.g. ":0")
            base = base.rsplit(":", 1)[0]

        if base == "embeddings" and "embedding" in lo:
            emb_w = data
        elif "forward" in lo:
            if base == "kernel" and fw_k is None:
                fw_k = data
            elif base == "recurrent_kernel":
                fw_rk = data
            elif base == "bias" and fw_b is None:
                fw_b = data
        elif "backward" in lo:
            if base == "kernel" and bw_k is None:
                bw_k = data
            elif base == "recurrent_kernel":
                bw_rk = data
            elif base == "bias" and bw_b is None:
                bw_b = data
        elif "dense" in lo and "forward" not in lo and "backward" not in lo:
            if base == "kernel" and d_k is None:
                d_k = data
            elif base == "bias" and d_b is None:
                d_b = data

    missing = [n for n, w in [
        ("embedding.embeddings", emb_w),
        ("forward_lstm.kernel", fw_k), ("forward_lstm.recurrent_kernel", fw_rk),
        ("forward_lstm.bias", fw_b),
        ("backward_lstm.kernel", bw_k), ("backward_lstm.recurrent_kernel", bw_rk),
        ("backward_lstm.bias", bw_b),
        ("dense.kernel", d_k), ("dense.bias", d_b),
    ] if w is None]
    if missing:
        raise RuntimeError(f"h5py fallback: could not locate weights: {missing}")

    input_dim, output_dim = emb_w.shape
    model = Sequential([
        Embedding(input_dim, output_dim, input_length=max_len),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(32, recurrent_dropout=0.2)),
        Dense(len(CLASSES), activation="softmax"),
    ])
    # Force a build so all layer weights are allocated before set_weights.
    model(np.zeros((1, max_len), dtype=np.int32), training=False)

    # layers: 0=Embedding, 1=SpatialDropout1D, 2=Bidirectional, 3=Dense
    model.layers[0].set_weights([emb_w])
    model.layers[2].set_weights([fw_k, fw_rk, fw_b, bw_k, bw_rk, bw_b])
    model.layers[3].set_weights([d_k, d_b])
    return model


def _load_bilstm():
    # Level 1: normal load_model path (fast; works when Keras versions match).
    _e1 = None
    try:
        from tensorflow.keras.models import load_model  # type: ignore
        model = load_model(str(BILSTM_H5), compile=False)
    except Exception as e:
        msg = str(e).lower()
        is_config_err = any(k in msg for k in (
            "unrecognized keyword", "quantization_config",
            "cannot deserialize", "deserialization",
        ))
        if not is_config_err:
            raise RuntimeError(f"BiLSTM load error: {e}") from e
        print(f"  [bilstm] load_model failed ({type(e).__name__}); using h5py fallback…")
        _e1 = e
        model = None

    # Level 2: h5py fallback (handles cross-version config-deserialization errors).
    if model is None:
        try:
            model = _bilstm_h5py_fallback()
        except Exception as e2:
            raise RuntimeError(
                f"BiLSTM load error: both load paths failed.\n"
                f"  load_model: {_e1}\n"
                f"  h5py fallback: {e2}"
            ) from e2

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def _load_ulmfit():
    try:
        from fastai.text.all import load_learner  # type: ignore
        # Models exported on Linux store PosixPath; patch for Windows loading only.
        _orig = pathlib.PosixPath if sys.platform == "win32" else None
        if sys.platform == "win32":
            pathlib.PosixPath = pathlib.WindowsPath
        try:
            learn = load_learner(str(ULMFIT_PKL))
        finally:
            if _orig is not None:
                pathlib.PosixPath = _orig
        return learn
    except Exception as e:
        raise RuntimeError(f"ULMFiT load error: {e}")


def _load_transformer_model(name: str):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
        import torch  # type: ignore
        path = str(TR_DIRS[name])
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        model.eval()
        model.to("cpu")
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"{name} load error: {e}")


def _get_model(name: str):
    with _model_lock:
        if name in _model_cache:
            return _model_cache[name]
        if not _model_file_ok(name):
            return None
        if name == "BiLSTM":
            obj = _load_bilstm()
        elif name == "ULMFiT":
            obj = _load_ulmfit()
        elif name in TR_MODELS:
            obj = _load_transformer_model(name)
        else:
            return None
        _model_cache[name] = obj
        return obj


def _predict_bilstm(text: str) -> np.ndarray:
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
    model, tokenizer = _get_model("BiLSTM")
    cleaned = _apply_contractions(clean_tweet(text), prep_cfg)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=prep_cfg.get("max_len", 60),
                            padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]
    return np.array(probs, dtype=float)


def _predict_ulmfit(text: str) -> np.ndarray:
    learn = _get_model("ULMFiT")
    cleaned = clean_tweet(text)
    _, _, raw_probs = learn.predict(cleaned)
    raw_probs = raw_probs.numpy()
    # learn.dls.vocab is (text_vocab, label_vocab). label_vocab contains the
    # integer-encoded class indices in fastai's sort order (not class-name strings).
    # Use label_encoder to resolve CLASSES → int indices → positions in label_vocab.
    try:
        label_vocab = [int(x) for x in learn.dls.vocab[1]]
        if label_encoder is not None:
            canonical_indices = [int(label_encoder.transform([c])[0]) for c in CLASSES]
        else:
            canonical_indices = list(range(len(CLASSES)))
        reordered = np.array(
            [raw_probs[label_vocab.index(ci)] for ci in canonical_indices],
            dtype=float,
        )
        return reordered
    except Exception:
        return raw_probs.astype(float)


def _predict_transformer(text: str, name: str) -> np.ndarray:
    import torch  # type: ignore
    model, tokenizer = _get_model(name)
    cleaned = clean_tweet(text)
    inputs  = tokenizer(
        cleaned, return_tensors="pt", truncation=True,
        max_length=256, padding=True,
    )
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].numpy().astype(float)
    return probs


def run_inference(text: str, model_name: str) -> tuple[dict, float]:
    obj = _get_model(model_name)
    if obj is None:
        return {}, -1.0
    t0 = time.perf_counter()
    try:
        if model_name == "BiLSTM":
            probs = _predict_bilstm(text)
        elif model_name == "ULMFiT":
            probs = _predict_ulmfit(text)
        else:
            probs = _predict_transformer(text, model_name)
    except Exception as e:
        print(f"  [inference error] {model_name}: {e}")
        return {}, -1.0
    elapsed = (time.perf_counter() - t0) * 1000
    probs_dict = {c: float(p) for c, p in zip(CLASSES, probs)}
    return probs_dict, round(elapsed, 1)

# ══════════════════════════════════════════════════════════════════════════════
# GROQ EXPLANATION
# ══════════════════════════════════════════════════════════════════════════════

def _clean_dashes(text: str) -> str:
    """Replace unicode em/en dashes with plain ASCII hyphen for clean display."""
    if not isinstance(text, str):
        return text
    return (text
            .replace("—", "-")
            .replace("–", "-")
            .replace("‒", "-")
            .replace("−", "-"))


def get_groq_reference(tweet: str, api_key: str = "") -> tuple[str, str] | None:
    """Phase 1: assess the tweet's actual sentiment ONCE, independent of any model.

    Returns (reference_class, justification) or None when the package / key is
    unavailable.  Returns ("[Groq error: …]", "") on API/network failure.
    """
    key = api_key.strip() or GROQ_API_KEY
    if not GROQ_PKG_OK or not key:
        return None

    system_msg = (
        "You are an independent analyst assessing tweets about the Artemis II lunar mission. "
        "The four sentiment classes are:\n"
        "  • Conspiratorial: denies mission authenticity (CGI, hoax, green screen, cover-up)\n"
        "  • Critical/Skeptical: questions the mission's value, cost, or execution; "
        "practical doubts, not denial\n"
        "  • Enthusiastic: positive, excited reactions to the mission\n"
        "  • Neutral: informational, news-style, no strong opinion\n\n"
        "Respond in EXACTLY this two-line format and nothing else:\n"
        "Class: <exact class name>\n"
        "Reason: <2-3 sentences explaining why this class fits best>\n"
        "Do not mention any model prediction. Judge the tweet on its own merits."
    )

    few_shot = [
        {
            "role": "user",
            "content": (
                'Tweet: "Just watched the Artemis II crew board the Orion capsule LIVE. '
                "Tears in my eyes! humanity's return to the Moon after 50 years! GO NASA!\""
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Class: Enthusiastic\n"
                "Reason: The tweet expresses strong personal emotion (\"tears in my eyes\") and "
                "uses all-caps celebration (\"GO NASA!\"). The exclamatory language and historical "
                "framing are unambiguous markers of enthusiastic positivity."
            ),
        },
        {
            "role": "user",
            "content": (
                'Tweet: "Notice how they cut the feed every time the camera points outside? '
                "Same CGI tricks as Apollo, green screens don't fool everyone. "
                'Wake up. #FakeArtemis"'
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Class: Conspiratorial\n"
                "Reason: The tweet alleges deliberate feed cuts and invokes the Apollo hoax "
                "narrative. CGI denial language and the #FakeArtemis hashtag are hallmark "
                "hoax-conspiracy signals."
            ),
        },
        {
            "role": "user",
            "content": (
                'Tweet: "Great, another $4 billion to orbit the Moon without landing. '
                "SpaceX does this cheaper. Hope this PR stunt actually leads somewhere this time.\""
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Class: Critical/Skeptical\n"
                "Reason: The sarcastic opener, explicit cost complaint, and 'PR stunt' label "
                "signal practical skepticism about the mission's value. There is no hoax "
                "denial, this is criticism, not conspiracy."
            ),
        },
        {
            "role": "user",
            "content": (
                'Tweet: "Artemis II splashed down safely. Crew recovery is underway. '
                'Total programme cost: approximately $4.1 billion."'
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Class: Neutral\n"
                "Reason: The tweet reports factual events without evaluative language. "
                "Even the cost figure is presented as a data point, not a criticism."
            ),
        },
    ]

    messages = (
        [{"role": "system", "content": system_msg}]
        + few_shot
        + [{"role": "user", "content": f'Tweet: "{tweet}"'}]
    )

    try:
        client = _Groq(api_key=key)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        raw = resp.choices[0].message.content.strip()
        ref_class = ""
        reason_parts: list[str] = []
        for line in raw.splitlines():
            s = line.strip()
            if s.startswith("Class:"):
                candidate = s[len("Class:"):].strip()
                for c in CLASSES:
                    if c.lower() == candidate.lower() or c.lower() in candidate.lower():
                        ref_class = c
                        break
                if not ref_class:
                    ref_class = candidate
            elif s.startswith("Reason:"):
                reason_parts.append(s[len("Reason:"):].strip())
            elif reason_parts:
                reason_parts.append(s)
        if not ref_class:
            for c in CLASSES:
                if c in raw:
                    ref_class = c
                    break
        justification = " ".join(p for p in reason_parts if p)
        return ref_class or "Unknown", _clean_dashes(justification or raw)
    except Exception as e:
        return f"[Groq error: {e}]", ""


def get_groq_comparison(tweet: str, reference_class: str, predicted_class: str,
                        api_key: str = "") -> str | None:
    """Phase 2: compare one model's prediction to the Phase-1 reference class.

    Returns a 2-3 sentence assessment string, None when unavailable, or a
    "[Groq error: …]" string on API/network failure.
    """
    key = api_key.strip() or GROQ_API_KEY
    if not GROQ_PKG_OK or not key:
        return None

    system_msg = (
        "You are reviewing an NLP model's sentiment prediction against an independent "
        "reference assessment. The reference class was determined independently and "
        "should be treated as the established assessment for this tweet.\n\n"
        "Respond in 2-3 sentences:\n"
        "• Prediction matches reference → confirm it is correct and briefly explain why "
        "the reference class fits.\n"
        "• Prediction differs from reference → state the prediction is wrong, name the "
        "reference class as the better fit, and explain why.\n"
        "Do not re-assess the tweet from scratch."
    )

    few_shot = [
        {
            "role": "user",
            "content": (
                'Tweet: "Just watched the Artemis II crew board the Orion capsule LIVE. '
                "Tears in my eyes, humanity's return to the Moon after 50 years! GO NASA!\"\n"
                "Reference class: Enthusiastic\n"
                "Model predicted: Enthusiastic"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "The model's prediction matches the reference: Enthusiastic is correct. "
                "The exclamatory language and personal emotional reaction are clear "
                "positive-sentiment markers consistent with the reference assessment."
            ),
        },
        {
            "role": "user",
            "content": (
                'Tweet: "Notice how they cut the feed every time the camera points outside? '
                "Same CGI tricks as Apollo, green screens don't fool everyone. "
                'Wake up. #FakeArtemis"\n'
                "Reference class: Conspiratorial\n"
                "Model predicted: Neutral"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "The model's prediction is wrong. The reference class Conspiratorial fits "
                "far better, the tweet invokes the Apollo hoax, uses CGI denial language, "
                "and includes #FakeArtemis, none of which are compatible with Neutral."
            ),
        },
        {
            "role": "user",
            "content": (
                'Tweet: "Great, another $4 billion to orbit the Moon without landing. '
                "SpaceX does this cheaper. Hope this PR stunt actually leads somewhere this time.\"\n"
                "Reference class: Critical/Skeptical\n"
                "Model predicted: Enthusiastic"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "The model's prediction is wrong. The reference class Critical/Skeptical "
                "fits far better, the sarcastic tone, cost complaint, and 'PR stunt' label "
                "are clear skepticism markers that contradict an Enthusiastic classification."
            ),
        },
        {
            "role": "user",
            "content": (
                'Tweet: "Artemis II splashed down safely. Crew recovery is underway. '
                'Total programme cost: approximately $4.1 billion."\n'
                "Reference class: Neutral\n"
                "Model predicted: Neutral"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "The model's prediction matches the reference: Neutral is correct. "
                "The tweet reports factual events without evaluative language, consistent "
                "with the reference assessment."
            ),
        },
    ]

    messages = (
        [{"role": "system", "content": system_msg}]
        + few_shot
        + [{"role": "user", "content": (
            f'Tweet: "{tweet}"\n'
            f"Reference class: {reference_class}\n"
            f"Model predicted: {predicted_class}"
        )}]
    )

    try:
        client = _Groq(api_key=key)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        return _clean_dashes(resp.choices[0].message.content.strip())
    except Exception as e:
        return f"[Groq error: {e}]"

# ══════════════════════════════════════════════════════════════════════════════
# UI HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

_CM_COLORSCALE = [
    [0.0,  "#152a5a"],
    [0.35, "#1a5090"],
    [0.65, "#1a8abf"],
    [0.85, "#4ab8e0"],
    [1.0,  "#b0e4f5"],
]


def confusion_matrix_fig(cm: np.ndarray, classes: list[str], title: str) -> go.Figure:
    cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100
    annotations = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            text_color = "#0a0f1e" if cm_pct[i, j] > 55 else "#e2e8f0"
            annotations.append(dict(
                x=j, y=i,
                text=f"<b>{cm[i, j]}</b><br>{cm_pct[i, j]:.1f}%",
                showarrow=False, font=dict(color=text_color, size=14),
                xref="x", yref="y",
            ))
    fig = go.Figure(go.Heatmap(
        z=cm_pct, x=classes, y=classes,
        colorscale=_CM_COLORSCALE, showscale=True, zmin=0, zmax=100,
        colorbar=dict(
            title="%", ticksuffix="%",
            tickfont=dict(size=12, color="#e2e8f0"),
            title_font=dict(color="#e2e8f0"),
        ),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor="#0f1629", plot_bgcolor="#0f1629",
        title_text=title,
        xaxis_title="Predicted", yaxis_title="True",
        xaxis=dict(tickfont=dict(size=12), title_font=dict(size=13)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12), title_font=dict(size=13)),
        annotations=annotations,
        height=520, margin=dict(l=10, r=20, t=60, b=10),
    )
    return fig


def classification_report_table(report: dict) -> dbc.Table:
    rows = []
    for cls in CLASSES:
        r = report.get(cls, {})
        rows.append(html.Tr([
            html.Td(cls, style={"color": CLASS_COLORS.get(cls, "#e2e8f0"), "fontWeight": "600"}),
            html.Td(f"{r.get('precision', 0):.4f}"),
            html.Td(f"{r.get('recall', 0):.4f}"),
            html.Td(f"{r.get('f1-score', 0):.4f}"),
            html.Td(str(int(r.get("support", 0)))),
        ]))
    macro = report.get("macro avg", {})
    rows.append(html.Tr([
        html.Td("Macro Avg", style={"fontStyle": "italic", "color": "#94a3b8"}),
        html.Td(f"{macro.get('precision', 0):.4f}"),
        html.Td(f"{macro.get('recall', 0):.4f}"),
        html.Td(f"{macro.get('f1-score', 0):.4f}"),
        html.Td("N/A"),
    ], style={"borderTop": "1px solid #1e3a5f"}))
    return dbc.Table(
        [html.Thead(html.Tr([
            html.Th("Class"), html.Th("Precision"),
            html.Th("Recall"), html.Th("F1"), html.Th("Support"),
        ])), html.Tbody(rows)],
        bordered=True, hover=True, responsive=True,
        className="table-dark", style={"fontSize": "0.88rem"},
    )


def model_comparison_table() -> dbc.Table:
    rows = []
    for name in MODEL_NAMES:
        m   = model_metrics.get(name, {})
        lat = latency_data.get(name, {})
        rows.append(html.Tr([
            html.Td(html.B(name)),
            html.Td(f"{m.get('macro_f1', 'N/A')}"),
            html.Td(f"{m.get('macro_prec', 'N/A')}"),
            html.Td(f"{m.get('macro_rec', 'N/A')}"),
            html.Td(f"{m.get('conspir_f1', 'N/A')}"),
            html.Td(f"{m.get('critical_f1', 'N/A')}"),
            html.Td(f"{lat.get('latency_ms', 'N/A')}"),
            html.Td(f"{lat.get('size_mb', 'N/A')}"),
        ]))
    return dbc.Table(
        [html.Thead(html.Tr([
            html.Th("Model"), html.Th("Macro F1"), html.Th("Macro Prec."),
            html.Th("Macro Rec."), html.Th("Conspiratorial F1"),
            html.Th("Critical/Skept. F1"), html.Th("CPU Latency (ms)"),
            html.Th("Size (MB)"),
        ])), html.Tbody(rows)],
        bordered=True, hover=True, responsive=True,
        className="table-dark", style={"fontSize": "0.85rem"},
    )


def comparison_bubble_fig() -> go.Figure:
    rows = []
    for name in MODEL_NAMES:
        m   = model_metrics.get(name, {})
        lat = latency_data.get(name, {})
        if not m or not lat:
            continue
        rows.append({
            "model":    name,
            "macro_f1": m["macro_f1"],
            "latency":  lat["latency_ms"],
            "size":     lat["size_mb"],
        })
    if not rows:
        return _empty_fig("No comparison data available.")
    df = pd.DataFrame(rows)
    s_min, s_max = 120, 1600
    sz    = df["size"]
    bubble = s_min + (sz - sz.min()) / (sz.max() - sz.min() + 1e-9) * (s_max - s_min)

    fig = go.Figure(go.Scatter(
        x=df["latency"], y=df["macro_f1"],
        mode="markers+text",
        text=df["model"] + "<br>(" + df["size"].astype(str) + " MB)",
        textposition="top center",
        marker=dict(
            size=np.sqrt(bubble) / 1.5,
            color=df["macro_f1"],
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="Macro F1"),
            line=dict(color="white", width=1),
            opacity=0.8,
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Latency: %{x:.1f} ms<br>"
            "Macro F1: %{y:.4f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor="#0f1629", plot_bgcolor="#0f1629",
        title_text="Quality vs. Latency vs. Size Trade-off",
        xaxis_title="CPU Latency (ms / tweet)",
        yaxis_title="Macro F1",
        height=420, margin=dict(t=50, b=50, l=50, r=10),
    )
    return fig


FIG_COMPARISON_BUBBLE = comparison_bubble_fig()


def prob_bar_chart(probs_dict: dict, title: str) -> go.Figure:
    items  = sorted(probs_dict.items(), key=lambda x: -x[1])
    classes = [i[0] for i in items]
    values  = [i[1] for i in items]
    colors  = [CLASS_COLORS.get(c, "#00d4ff") for c in classes]
    fig = go.Figure(go.Bar(
        x=values, y=classes, orientation="h",
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor="#0f1629", plot_bgcolor="#0f1629",
        title_text=title, xaxis_range=[0, 1.05],
        xaxis_tickformat=".0%",
        height=220, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def model_result_card(model_name: str, probs_dict: dict, elapsed_ms: float) -> html.Div:
    if not probs_dict:
        return dbc.Alert(
            [
                html.B(model_name), ": model not available. ",
                html.Span("Run python download_models.py to download model weights.",
                          style={"fontSize": "0.85em"}),
            ],
            color="warning", style={"marginBottom": "8px"},
        )
    predicted = max(probs_dict, key=probs_dict.get)
    color = CLASS_COLORS.get(predicted, "#00d4ff")
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Span(model_name, style={"fontWeight": "700", "color": "#00d4ff"}),
                html.Span("  →  ", style={"color": "#94a3b8"}),
                html.Span(predicted, style={"fontWeight": "700", "color": color}),
                html.Span(
                    f"  ({probs_dict[predicted]:.1%})",
                    style={"color": "#94a3b8", "fontSize": "0.9em"},
                ),
            ], width=8),
            dbc.Col(
                html.Span(f"{elapsed_ms:.0f} ms", className="timing-badge"),
                width=4, style={"textAlign": "right"},
            ),
        ], align="center", className="mb-2"),
        dcc.Graph(
            figure=prob_bar_chart(probs_dict, ""),
            config={"displayModeBar": False},
            style={"height": "180px"},
        ),
    ], style={
        "background": "#0f1629",
        "border": f"1px solid {color}40",
        "borderLeft": f"3px solid {color}",
        "borderRadius": "8px",
        "padding": "12px",
        "marginBottom": "12px",
    })


def _live_model_block(model_name: str, probs_dict: dict, elapsed_ms: float,
                      comparison_text: str | None, api_key_provided: bool) -> html.Div:
    """Full card for the Live Test panel: prediction bar + Phase-2 per-model comparison."""
    if not probs_dict:
        return dbc.Alert(
            [html.B(model_name), ": model not available. ",
             html.Span("Run python download_models.py to download model weights.",
                       style={"fontSize": "0.85em"})],
            color="warning", style={"marginBottom": "20px"},
        )

    predicted = max(probs_dict, key=probs_dict.get)
    color = CLASS_COLORS.get(predicted, "#00d4ff")

    # ── Phase-2 comparison node ────────────────────────────────────────────────
    if comparison_text is None:
        if not GROQ_PKG_OK:
            cmp_node = html.P(
                "groq package not installed - run: pip install groq",
                style={"color": "#94a3b8", "fontSize": "0.85rem",
                       "fontStyle": "italic", "margin": 0},
            )
        elif not api_key_provided:
            cmp_node = html.P(
                "Enter your Groq API key in the left panel to enable LLM explanations.",
                style={"color": "#64748b", "fontSize": "0.85rem",
                       "fontStyle": "italic", "margin": 0},
            )
        else:
            cmp_node = html.P(
                "LLM comparison unavailable.",
                style={"color": "#94a3b8", "fontSize": "0.88rem", "margin": 0},
            )
    elif comparison_text.startswith("[Groq error:"):
        cmp_node = dbc.Alert(comparison_text, color="danger",
                             style={"fontSize": "0.82rem", "marginBottom": 0})
    else:
        cmp_node = html.P(
            comparison_text,
            style={"color": "#e2e8f0", "lineHeight": "1.75",
                   "fontSize": "0.92rem", "margin": 0},
        )

    return html.Div([
        # Row 1: model name + timing badge
        dbc.Row([
            dbc.Col(
                html.Span(model_name,
                          style={"fontWeight": "700", "color": "#00d4ff",
                                 "fontSize": "1.05rem"}),
                width="auto",
            ),
            dbc.Col(
                html.Span(f"{elapsed_ms:.0f} ms", className="timing-badge"),
                width="auto",
            ),
        ], align="center", justify="between", className="mb-1"),

        # Row 2: predicted label line
        html.Div([
            html.Span("→ ", style={"color": "#94a3b8"}),
            html.Span(predicted, style={"color": color, "fontWeight": "700"}),
            html.Span(f"  ({probs_dict[predicted]:.1%})",
                      style={"color": "#94a3b8", "fontSize": "0.9em"}),
        ], style={"marginBottom": "6px"}),

        # Row 3: probability bar chart
        dcc.Graph(
            figure=prob_bar_chart(probs_dict, ""),
            config={"displayModeBar": False},
            style={"height": "160px"},
        ),

        # Row 4: Phase-2 per-model comparison against the reference
        html.Div([
            html.Hr(style={"borderColor": "#1e3a5f", "margin": "10px 0"}),
            html.Div([
                html.Span("vs. LLM Reference",
                          style={"color": "#a78bfa", "fontWeight": "700",
                                 "fontSize": "0.75rem", "textTransform": "uppercase",
                                 "letterSpacing": "0.06em"}),
                html.Span(f"  ·  Groq / {GROQ_MODEL}",
                          style={"color": "#475569", "fontSize": "0.72rem"}),
            ], style={"marginBottom": "8px"}),
            cmp_node,
        ]),
    ], style={
        "background": "#0f1629",
        "border": "1px solid #1e3a5f",
        "borderLeft": f"3px solid {color}",
        "borderRadius": "10px",
        "padding": "16px 18px",
        "marginBottom": "20px",
    })


# ══════════════════════════════════════════════════════════════════════════════
# SVG ICONS
# ══════════════════════════════════════════════════════════════════════════════

_moon_svg = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20">'
    '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"'
    ' stroke="#00d4ff" stroke-width="2" stroke-linecap="round"'
    ' stroke-linejoin="round" fill="none"/></svg>'
)
_moon_icon = html.Img(
    src="data:image/svg+xml;base64," + base64.b64encode(_moon_svg.encode()).decode(),
    style={"verticalAlign": "middle", "marginRight": "8px", "display": "inline-block",
           "width": "20px", "height": "20px"},
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# ── HOME ──────────────────────────────────────────────────────────────────────

def _stat_card(value, label, color="#00d4ff"):
    return html.Div([
        html.Div(str(value), className="stat-value", style={"color": color}),
        html.Div(label, className="stat-label"),
    ], className="stat-badge")


# ── GitHub Octocat icon (official GitHub mark, Base64 SVG data URI) ──────────
_github_svg = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 98 96"'
    ' width="24" height="24">'
    '<path fill="#94a3b8" fill-rule="evenodd" clip-rule="evenodd" d="'
    'M48.854 0C21.839 0 0 22 0 49.217c0 21.756 13.993 40.172 33.405 46.69'
    ' 2.427.49 3.316-1.059 3.316-2.362 0-1.141-.08-5.052-.08-9.127'
    '-13.59 2.934-16.42-5.867-16.42-5.867-2.184-5.704-5.42-7.17-5.42-7.17'
    '-4.448-3.015.324-3.015.324-3.015 4.934.326 7.523 5.052 7.523 5.052'
    ' 4.367 7.496 11.404 5.378 14.235 4.074.404-3.178 1.699-5.378 3.074-6.6'
    '-10.839-1.141-22.243-5.378-22.243-24.283 0-5.378 1.94-9.778 5.014-13.2'
    '-.485-1.222-2.184-6.275.486-13.038 0 0 4.125-1.304 13.426 5.052'
    ' a46.97 46.97 0 0 1 12.214-1.63c4.125 0 8.33.571 12.213 1.63'
    ' 9.302-6.356 13.427-5.052 13.427-5.052 2.67 6.763.97 11.816.485 13.038'
    ' 3.155 3.422 5.015 7.822 5.015 13.2 0 18.905-11.404 23.06-22.324 24.283'
    ' 1.78 1.548 3.316 4.481 3.316 9.126 0 6.6-.08 11.897-.08 13.526'
    ' 0 1.304.89 2.853 3.316 2.364 19.412-6.52 33.405-24.935 33.405-46.691'
    'C97.707 22 75.788 0 48.854 0z"/></svg>'
)
_github_icon_uri = (
    "data:image/svg+xml;base64,"
    + base64.b64encode(_github_svg.encode()).decode()
)
_GITHUB_MEMBERS = [
    ("Mirko Dervishi",  "https://github.com/Mirko-hubgit"),
    ("Matteo Gerevini", "https://github.com/00gerem00"),
    ("Andrea Grulla",   "https://github.com/grullaandrea-png"),
    ("Lorenzo Meroni",  "https://github.com/lorenzomeroni02"),
]

home_content = dbc.Container([
    # ── Header: title + GitHub icon ──────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1(
                    "ARTEMIS II - DATAUNDERDOGS",
                    style={
                        "fontSize": "2.2rem", "fontWeight": "800",
                        "letterSpacing": "0.02em", "color": "#e2e8f0",
                        "marginBottom": "0", "marginRight": "14px",
                    },
                ),
                html.Span([
                    html.Button(
                        html.Img(
                            src=_github_icon_uri,
                            style={"width": "26px", "height": "26px",
                                   "verticalAlign": "middle"},
                        ),
                        id="home-github-btn",
                        n_clicks=0,
                        title="Team GitHub profiles",
                        style={
                            "background": "none", "border": "none",
                            "cursor": "pointer", "padding": "4px",
                            "lineHeight": "1",
                        },
                    ),
                    dbc.Modal(
                        [
                            dbc.ModalHeader(
                                dbc.ModalTitle(
                                    "DATAUNDERDOGS",
                                    style={"color": "#00d4ff", "fontWeight": "700",
                                           "letterSpacing": "0.06em",
                                           "fontSize": "1rem"},
                                ),
                                close_button=True,
                                style={"background": "#0f1629",
                                       "borderBottom": "1px solid #1e3a5f"},
                            ),
                            dbc.ModalBody(
                                html.Ul([
                                    html.Li(
                                        html.A(
                                            name, href=url, target="_blank",
                                            style={"color": "#94a3b8",
                                                   "textDecoration": "none",
                                                   "fontSize": "0.95rem"},
                                        ),
                                        style={"padding": "6px 0",
                                               "listStyle": "none"},
                                    )
                                    for name, url in _GITHUB_MEMBERS
                                ], style={"padding": "0", "margin": "0"}),
                                style={"background": "#0f1629"},
                            ),
                        ],
                        id="github-members-modal",
                        is_open=False,
                        size="sm",
                        keyboard=True,
                        backdrop=True,
                    ),
                ]),
            ], style={"display": "flex", "alignItems": "center",
                      "flexWrap": "wrap", "gap": "4px"}),
            html.P(
                "Tweet Sentiment Analysis",
                style={"color": "#94a3b8", "fontSize": "1.05rem",
                       "fontWeight": "400", "marginTop": "6px",
                       "marginBottom": "0"},
            ),
        ], width=12),
    ], className="mb-5",
       style={"borderBottom": "1px solid #1e3a5f", "paddingBottom": "28px"}),

    # ── About Artemis II ──────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Div("About Artemis II", className="section-header"),
            html.P(
                "The Artemis program is NASA's effort to return humans to the Moon "
                "and build a sustained lunar presence as a stepping stone toward Mars. "
                "Artemis II (April 1-11, 2026) was the first crewed mission of the "
                "Artemis program: a crewed lunar flyby, the first crewed flight beyond "
                "low Earth orbit since Apollo 17 (1972), and the first crewed flight of "
                "the SLS rocket and the Orion spacecraft (named Integrity). It launched "
                "on April 1, 2026 from Kennedy Space Center (Florida) and splashed down "
                "in the Pacific Ocean on April 10, 2026, an approximately 10-day mission.",
                style={"color": "#e2e8f0", "lineHeight": "1.75",
                       "marginBottom": "14px"},
            ),
            html.P(
                "The crew comprised commander Reid Wiseman, pilot Victor Glover, and "
                "mission specialists Christina Koch and Jeremy Hansen (CSA). The main "
                "goal was to validate Orion's systems, life-support, crew operations, "
                "and procedures ahead of future crewed lunar missions. As a milestone, "
                "it carried the first astronauts to the Moon's vicinity since 1972, and "
                "the crew set the record for the farthest humans have traveled from Earth "
                "(approximately 406,771 km), surpassing Apollo 13.",
                style={"color": "#e2e8f0", "lineHeight": "1.75",
                       "marginBottom": "14px"},
            ),
            html.P(
                "Its key phases were launch (Apr 1), trans-lunar injection (Apr 2), a "
                "lunar flyby approximately 6,545 km from the Moon (Apr 6) with roughly "
                "40 minutes of far-side signal blackout, and splashdown (Apr 10), "
                "following a free-return figure-eight trajectory.",
                style={"color": "#e2e8f0", "lineHeight": "1.75",
                       "marginBottom": "0"},
            ),
        ], width=12),
    ], className="mb-5"),

    # ── Team + Dataset ────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Div("The Team", className="section-header"),
            html.P(
                "Mirko Dervishi, Matteo Gerevini, Andrea Grulla, Lorenzo Meroni",
                style={"color": "#e2e8f0", "fontSize": "1rem", "lineHeight": "1.6"},
            ),
        ], md=4, className="mb-4"),
        dbc.Col([
            html.Div("Dataset", className="section-header"),
            html.P(
                "6,624 annotated tweets, split by stratified sampling "
                "(class proportions preserved across all splits):",
                style={"color": "#e2e8f0", "lineHeight": "1.7",
                       "marginBottom": "10px"},
            ),
            dbc.Table(
                [
                    html.Thead(
                        html.Tr([
                            html.Th("Split"), html.Th("Tweets"), html.Th("Share"),
                        ]),
                        style={"color": "#00d4ff"},
                    ),
                    html.Tbody([
                        html.Tr([html.Td("Train"),      html.Td("4,636"), html.Td("70%")]),
                        html.Tr([html.Td("Validation"), html.Td("993"),   html.Td("15%")]),
                        html.Tr([html.Td("Test"),       html.Td("994"),   html.Td("15%")]),
                    ]),
                ],
                bordered=True, size="sm",
                className="table-dark",
                style={"maxWidth": "300px", "fontSize": "0.88rem"},
            ),
        ], md=8, className="mb-4"),
    ]),

    # ── Sentiment categories ──────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Div("Sentiment Categories", className="section-header"),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.Span("Conspiratorial",
                                  style={"fontWeight": "700", "color": "#ef4444",
                                         "fontSize": "0.95rem"}),
                        html.P("Sentiment that denies the mission's authenticity, "
                               "framing it as staged or fabricated.",
                               style={"color": "#94a3b8", "fontSize": "0.88rem",
                                      "lineHeight": "1.5", "marginTop": "6px",
                                      "marginBottom": "0"}),
                    ], style={**CARD_STYLE, "borderLeft": "3px solid #ef4444"}),
                    md=3, className="mb-3",
                ),
                dbc.Col(
                    html.Div([
                        html.Span("Critical/Skeptical",
                                  style={"fontWeight": "700", "color": "#f97316",
                                         "fontSize": "0.95rem"}),
                        html.P("Sentiment that questions the mission's value, cost, "
                               "or execution with practical criticism.",
                               style={"color": "#94a3b8", "fontSize": "0.88rem",
                                      "lineHeight": "1.5", "marginTop": "6px",
                                      "marginBottom": "0"}),
                    ], style={**CARD_STYLE, "borderLeft": "3px solid #f97316"}),
                    md=3, className="mb-3",
                ),
                dbc.Col(
                    html.Div([
                        html.Span("Enthusiastic",
                                  style={"fontWeight": "700", "color": "#22c55e",
                                         "fontSize": "0.95rem"}),
                        html.P("Positive sentiment expressing excitement or "
                               "strong support for the mission.",
                               style={"color": "#94a3b8", "fontSize": "0.88rem",
                                      "lineHeight": "1.5", "marginTop": "6px",
                                      "marginBottom": "0"}),
                    ], style={**CARD_STYLE, "borderLeft": "3px solid #22c55e"}),
                    md=3, className="mb-3",
                ),
                dbc.Col(
                    html.Div([
                        html.Span("Neutral",
                                  style={"fontWeight": "700", "color": "#3b82f6",
                                         "fontSize": "0.95rem"}),
                        html.P("Informational or news-style sentiment with no "
                               "strong positive or negative stance.",
                               style={"color": "#94a3b8", "fontSize": "0.88rem",
                                      "lineHeight": "1.5", "marginTop": "6px",
                                      "marginBottom": "0"}),
                    ], style={**CARD_STYLE, "borderLeft": "3px solid #3b82f6"}),
                    md=3, className="mb-3",
                ),
            ]),
        ], width=12),
    ], className="mb-5"),

    # ── Data collection + Models ──────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Div("Data Collection", className="section-header"),
            html.P(
                "Tweets were scraped from X (Twitter) using Apify scrapers "
                "across 5 targeted collection windows.",
                style={"color": "#e2e8f0", "lineHeight": "1.7"},
            ),
        ], md=6),
        dbc.Col([
            html.Div("Models", className="section-header"),
            html.Ul(
                [html.Li(m, style={"color": "#e2e8f0", "padding": "3px 0"})
                 for m in ["BiLSTM", "ULMFiT", "DistilBERT",
                           "RoBERTa", "DeBERTa-v3"]],
                style={"paddingLeft": "18px", "listStyleType": "disc"},
            ),
        ], md=6),
    ], className="mb-4"),
], fluid=True, style={"paddingBottom": "40px"})


# ── DATASET ────────────────────────────────────────────────────────────────────

prep_steps = [
    ("1. Raw Data Merge", "Five CSV files (flyby, return, departure, conspiracyhunt, photoday) loaded and concatenated. Each row tagged with its source file. Total: 11,969 rows."),
    ("2. Drop Missing Labels", "Rows without a Sentiment_label (NaN) removed. Remaining: 6,671 labeled rows. Shorthand labels mapped to full names: E→Enthusiastic, N→Neutral, C→Conspiratorial, S→Critical/Skeptical."),
    ("3. Text Cleaning", "Applied sequentially: (A) Mojibake fix (Latin-1 → UTF-8 decode attempt); (B) Emoji demojization, preserving semantic content; (C) URL removal; (D) '@' symbol removal (keeps username); (E) '#' removal (keeps text); (F) HTML entity removal (&amp; etc.); (G) Whitespace normalization."),
    ("4. Exact-Duplicate Removal", "12 exact duplicate tweets (matching on raw text) removed. Remaining: 6,659."),
    ("5. Near-Duplicate Removal", "Tweets masked by replacing @mentions→<USER> and URLs→<URL>, then lowercased. 35 near-duplicates removed. Remaining: 6,624."),
    ("6. Final Formatting", "Underscores in demojized text (e.g. folded_hands) replaced with spaces; whitespace re-normalized."),
    ("7. Output", "Final dataset saved as data/processed/artemis_master_dataset.csv with columns: text (raw), Sentiment_label, source, cleaned_text."),
]

_prep_rows = [
    html.Tr([
        html.Td(step, style={"fontWeight": "600", "color": "#00d4ff", "whiteSpace": "nowrap",
                              "verticalAlign": "top", "paddingRight": "16px"}),
        html.Td(desc, style={"color": "#e2e8f0", "fontSize": "0.88rem", "lineHeight": "1.6"}),
    ]) for step, desc in prep_steps
]

if df_master is not None:
    tbl_columns = [
        {"name": c, "id": c}
        for c in ["Sentiment_label", "source", "text", "cleaned_text"]
        if c in df_master.columns
    ]
    # Class-quota stratified sample: fixed quotas per sentiment class (minority
    # classes intentionally oversampled for exploration; source not used).
    _cls_quotas = {
        "Enthusiastic":       175,
        "Neutral":            175,
        "Critical/Skeptical": 75,
        "Conspiratorial":     75,
    }
    _tbl_frames = []
    for _cls, _quota in _cls_quotas.items():
        _grp = df_master[df_master["Sentiment_label"] == _cls]
        if len(_grp) > 0:
            _tbl_frames.append(_grp.sample(n=min(len(_grp), _quota), random_state=42))
    tbl_data = (
        pd.concat(_tbl_frames)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
        [[c["id"] for c in tbl_columns]]
        .to_dict("records")
    ) if _tbl_frames else []
else:
    tbl_columns, tbl_data = [], []

dataset_content = dbc.Container([
    dbc.Tabs([
        dbc.Tab(label="Data Preparation", tab_id="dp-prep", children=[
            html.Div(style={"height": "16px"}),
            html.Div("Data Preparation Pipeline", className="section-header"),
            dbc.Table(
                [html.Tbody(_prep_rows)],
                bordered=True, hover=True, responsive=True,
                className="table-dark", style={"fontSize": "0.88rem"},
            ),
        ]),
        dbc.Tab(label="Cleaned Dataset", tab_id="dp-data", children=[
            html.Div(style={"height": "16px"}),
            html.Div([
                html.H5("MASTER DATASET",
                        style={"color": "#00d4ff", "fontWeight": "700",
                               "letterSpacing": "0.08em", "marginBottom": "2px"}),
                html.P("random 500-tweet sample",
                       style={"color": "#94a3b8", "fontSize": "0.88rem",
                              "marginBottom": "14px"}),
            ]),
            dash_table.DataTable(
                id="master-table",
                columns=tbl_columns,
                data=tbl_data,
                filter_action="native",
                sort_action="native",
                page_action="native",
                page_size=20,
                style_table={"overflowX": "auto"},
                style_cell={
                    "backgroundColor": "#0f1629",
                    "color": "#e2e8f0",
                    "border": "1px solid #1e3a5f",
                    "padding": "8px 12px",
                    "fontSize": "0.82rem",
                    "textAlign": "left",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                    "whiteSpace": "nowrap",
                    "cursor": "pointer",
                },
                style_cell_conditional=[
                    {"if": {"column_id": "Sentiment_label"}, "maxWidth": "160px", "minWidth": "120px"},
                    {"if": {"column_id": "source"},          "maxWidth": "120px", "minWidth": "90px"},
                    {"if": {"column_id": "text"},            "maxWidth": "320px"},
                    {"if": {"column_id": "cleaned_text"},    "maxWidth": "320px"},
                ],
                style_header={
                    "backgroundColor": "#1a2540",
                    "color": "#00d4ff",
                    "fontWeight": "600",
                    "border": "1px solid #1e3a5f",
                    "fontSize": "0.82rem",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#111827"},
                    {"if": {"filter_query": '{Sentiment_label} = "Enthusiastic"'},       "color": "#22c55e"},
                    {"if": {"filter_query": '{Sentiment_label} = "Neutral"'},             "color": "#3b82f6"},
                    {"if": {"filter_query": '{Sentiment_label} = "Critical/Skeptical"'}, "color": "#f97316"},
                    {"if": {"filter_query": '{Sentiment_label} = "Conspiratorial"'},     "color": "#ef4444"},
                    {"if": {"state": "active"},
                     "backgroundColor": "#1a2540", "border": "1px solid #00d4ff"},
                ],
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(
                        dbc.ModalTitle(
                            "Tweet Detail",
                            style={"color": "#e2e8f0", "fontWeight": "700"},
                        ),
                        close_button=False,
                        style={"background": "#0f1629",
                               "borderBottom": "1px solid #1e3a5f"},
                    ),
                    dbc.ModalBody(
                        id="tweet-detail-modal-body",
                        style={"background": "#0a0f1e"},
                    ),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Close",
                            id="tweet-modal-close",
                            size="sm",
                            outline=True,
                            n_clicks=0,
                            style={"color": "#94a3b8",
                                   "borderColor": "#1e3a5f"},
                        ),
                        style={"background": "#0f1629",
                               "borderTop": "1px solid #1e3a5f",
                               "justifyContent": "flex-end"},
                    ),
                ],
                id="tweet-detail-modal",
                is_open=False,
                size="xl",
                scrollable=True,
                keyboard=True,
            ),
        ]),
    ], id="dataset-inner-tabs", active_tab="dp-prep"),
], fluid=True, style={"paddingBottom": "40px"})


# ── EDA ────────────────────────────────────────────────────────────────────────

def _summary_stats_table():
    if df_eda is None:
        return html.P("Dataset not available.", style={"color": "#94a3b8"})
    df = df_eda.copy()
    df["char_count"] = df["cleaned_text"].astype(str).apply(len)
    df["word_count"] = df["cleaned_text"].astype(str).apply(lambda x: len(x.split()))
    rows = []
    for col, label in [("char_count", "Character Count"), ("word_count", "Word Count")]:
        s = df[col].describe()
        rows.append(html.Tr([
            html.Td(label, style={"fontWeight": "600", "color": "#00d4ff"}),
            html.Td(f"{s['mean']:.1f}"), html.Td(f"{s['50%']:.0f}"),
            html.Td(f"{s['std']:.1f}"),  html.Td(f"{s['min']:.0f}"),
            html.Td(f"{s['max']:.0f}"),
        ]))
    return dbc.Table(
        [html.Thead(html.Tr([
            html.Th("Metric"), html.Th("Mean"), html.Th("Median"),
            html.Th("Std"), html.Th("Min"), html.Th("Max"),
        ])), html.Tbody(rows)],
        bordered=True, hover=True, responsive=True,
        className="table-dark", style={"fontSize": "0.85rem"},
    )


eda_content = dbc.Container([
    dbc.Tabs([
        # ── 3a Basic Statistics ─────────────────────────────────────────────
        dbc.Tab(label="Basic Statistics & Distribution", tab_id="eda-basic", children=[
            html.Div(style={"height": "16px"}),
            html.Div("Text Length Summary", className="section-header"),
            _summary_stats_table(),
            html.Div(style={"height": "16px"}),
            dcc.Graph(figure=FIG_LABEL_DIST, config={"displayModeBar": False}),
            html.Div(style={"height": "16px"}),
            dcc.Graph(figure=FIG_PHASE, config={"displayModeBar": False}),
            html.Div(style={"height": "16px"}),
            dcc.Graph(figure=FIG_LENGTH, config={"displayModeBar": False}),
        ]),

        # ── 3b Linguistics / NER ────────────────────────────────────────────
        dbc.Tab(label="Linguistics & NER", tab_id="eda-ling", children=[
            html.Div(style={"height": "16px"}),
            dcc.Loading(
                html.Div(id="eda-ling-content"),
                type="circle", color="#00d4ff",
            ),
        ]),

        # ── 3c TF-IDF ───────────────────────────────────────────────────────
        dbc.Tab(label="TF-IDF Features", tab_id="eda-tfidf", children=[
            html.Div(style={"height": "16px"}),
            dbc.Alert(
                "TF-IDF computed at runtime from cleaned_text via scikit-learn (no spaCy required).",
                color="info", style={"fontSize": "0.85rem"},
            ),
            dcc.Graph(figure=FIG_TFIDF,  config={"displayModeBar": False}),
            html.Div(style={"height": "16px"}),
            dcc.Graph(figure=FIG_BIGRAM, config={"displayModeBar": False}),
        ]),

        # ── 3d Word Clouds ──────────────────────────────────────────────────
        dbc.Tab(label="Word Clouds", tab_id="eda-wc", children=[
            html.Div(style={"height": "16px"}),
            dbc.Alert(
                "Word clouds are generated in memory from cleaned_text (domain anchors excluded). "
                + ("Generated on first view and cached for the session."
                   if WC_OK else
                   "Install the wordcloud library (pip install wordcloud) to enable this section."),
                color="info" if WC_OK else "warning",
                style={"fontSize": "0.85rem"},
            ),
            dcc.Loading(
                html.Div(id="wc-content"),
                type="circle", color="#00d4ff",
            ),
        ]),
    ], id="eda-inner-tabs", active_tab="eda-basic"),
], fluid=True, style={"paddingBottom": "40px"})


# ── MODELS ─────────────────────────────────────────────────────────────────────

def _model_report_section(model_name: str) -> html.Div:
    m = model_metrics.get(model_name)
    if m is None:
        return dbc.Alert(
            f"{model_name} probs file not found. "
            "Ensure results/{bilstm,ulmfit,transformers}/probs_*.npy are present.",
            color="warning",
        )
    fig_cm = confusion_matrix_fig(m["cm"], CLASSES, f"{model_name} - Confusion Matrix")
    report_tbl = classification_report_table(m["report"])
    badge = html.Span(
        f"Macro F1: {m['macro_f1']:.4f}",
        style={"background": "rgba(0,212,255,0.15)", "color": "#00d4ff",
               "borderRadius": "12px", "padding": "4px 12px",
               "fontSize": "0.85rem", "fontWeight": "600"},
    )
    return html.Div([
        dbc.Row([
            dbc.Col(html.H5(model_name, style={"color": "#e2e8f0"}), width="auto"),
            dbc.Col(badge, width="auto"),
        ], align="center", className="mb-3"),
        dbc.Row([
            dbc.Col(report_tbl, width=12, lg=4),
            dbc.Col(dcc.Graph(figure=fig_cm, config={"displayModeBar": False}), width=12, lg=8),
        ]),
        html.Hr(style={"borderColor": "#1e3a5f"}),
    ], id=f"model-section-{model_name}")


# ── Balanced 100-tweet inference batch (fixed seed, reproducible) ─────────────
_BATCH_SEED     = 42
_BATCH_SIZE     = 100
_BATCH_MIN_MIN  = 20   # guaranteed minimum per minority class

_inference_batch: pd.DataFrame = pd.DataFrame()
_batch_table_data: list = []

if df_test is not None:
    try:
        _minority_cls = ["Conspiratorial", "Critical/Skeptical"]
        _majority_cls = ["Enthusiastic",   "Neutral"]
        _parts: list[pd.DataFrame] = []
        for _cls in _minority_cls:
            _pool = df_test[df_test["label"] == _cls]
            _n    = min(_BATCH_MIN_MIN, len(_pool))
            if _n:
                _parts.append(_pool.sample(n=_n, random_state=_BATCH_SEED))
        _used      = sum(len(p) for p in _parts)
        _remaining = _BATCH_SIZE - _used
        _per_maj   = max(1, _remaining // len(_majority_cls))
        for _cls in _majority_cls:
            _pool = df_test[df_test["label"] == _cls]
            _n    = min(_per_maj, len(_pool))
            if _n:
                _parts.append(_pool.sample(n=_n, random_state=_BATCH_SEED))
        if _parts:
            _inference_batch = (
                pd.concat(_parts)
                .sample(frac=1, random_state=_BATCH_SEED)
                .reset_index(drop=True)
            )
            _batch_table_data = [
                {
                    "row":     int(i),
                    "label":   str(row.get("label", "")),
                    "preview": (
                        str(row.get("text", ""))[:95] + "…"
                        if len(str(row.get("text", ""))) > 95
                        else str(row.get("text", ""))
                    ),
                }
                for i, row in _inference_batch.iterrows()
            ]
            print(
                f"[startup] Inference batch: {len(_inference_batch)} tweets: "
                + ", ".join(
                    f"{cls}={(_inference_batch['label'] == cls).sum()}"
                    for cls in _minority_cls + _majority_cls
                )
            )
    except Exception as _e:
        print(f"  [warn] inference batch build: {_e}")

models_content = dbc.Container([
    dbc.Tabs([
        # ── 4a Neural Networks ──────────────────────────────────────────────
        dbc.Tab(label="Neural Networks", tab_id="models-nn", children=[
            html.Div(style={"height": "16px"}),
            dbc.Alert(
                "Metrics recomputed from saved probability files (probs_*.npy) via scikit-learn. "
                "No model weights are loaded for this display.",
                color="info", style={"fontSize": "0.85rem"},
            ),
            html.Div(
                [
                    dbc.Button(
                        name, id=f"jump-{name}", n_clicks=0, size="sm",
                        style={
                            "marginRight": "8px", "marginBottom": "4px",
                            "background": "rgba(0,212,255,0.08)",
                            "border": "1px solid rgba(0,212,255,0.3)",
                            "color": "#00d4ff", "fontWeight": "600",
                            "fontSize": "0.8rem", "letterSpacing": "0.03em",
                        },
                    )
                    for name in ["BiLSTM", "ULMFiT"]
                ],
                style={"marginBottom": "18px"},
            ),
            _model_report_section("BiLSTM"),
            _model_report_section("ULMFiT"),
        ]),

        # ── 4b Transformers ─────────────────────────────────────────────────
        dbc.Tab(label="Transformers", tab_id="models-tr", children=[
            html.Div(style={"height": "16px"}),
            dbc.Alert(
                "Metrics recomputed from saved probability files (probs_*.npy) via scikit-learn. "
                "No model weights are loaded for this display.",
                color="info", style={"fontSize": "0.85rem"},
            ),
            html.Div(
                [
                    dbc.Button(
                        name, id=f"jump-{name}", n_clicks=0, size="sm",
                        style={
                            "marginRight": "8px", "marginBottom": "4px",
                            "background": "rgba(0,212,255,0.08)",
                            "border": "1px solid rgba(0,212,255,0.3)",
                            "color": "#00d4ff", "fontWeight": "600",
                            "fontSize": "0.8rem", "letterSpacing": "0.03em",
                        },
                    )
                    for name in ["DistilBERT", "RoBERTa", "DeBERTa-v3"]
                ],
                style={"marginBottom": "18px"},
            ),
            _model_report_section("DistilBERT"),
            _model_report_section("RoBERTa"),
            _model_report_section("DeBERTa-v3"),
        ]),

        # ── 4c Comparison ───────────────────────────────────────────────────
        dbc.Tab(label="Model Comparison", tab_id="models-cmp", children=[
            html.Div(style={"height": "16px"}),
            _download_notice(),
            html.Div("Full Comparison Table", className="section-header"),
            model_comparison_table(),
            html.Div(style={"height": "20px"}),
            dcc.Graph(figure=FIG_COMPARISON_BUBBLE, config={"displayModeBar": False}),
            html.Hr(style={"borderColor": "#1e3a5f"}),

            html.Div("Live Inference: Test-Set Tweet", className="section-header"),
            dbc.Alert(
                [
                    html.B("100-tweet balanced batch"),
                    " (seed=42, ≥20 per minority class). Click a tweet to select it, "
                    "or press ",
                    html.B("Tweet"),
                    " for a random pick, then ",
                    html.B("Classify All"),
                    " to run all 5 models. Requires model weights (",
                    html.Code("python download_models.py"),
                    ").",
                ],
                color="info", style={"fontSize": "0.85rem"},
            ),
            dbc.Row([
                # Left column: scrollable tweet table
                dbc.Col([
                    html.Div(
                        dash_table.DataTable(
                            id="cmp-tweet-table",
                            columns=[
                                {"name": "Class",         "id": "label"},
                                {"name": "Tweet Preview", "id": "preview"},
                            ],
                            data=_batch_table_data,
                            row_selectable="single",
                            selected_rows=[],
                            page_size=12,
                            style_table={
                                "overflowX": "auto",
                                "maxHeight": "440px",
                                "overflowY": "auto",
                            },
                            style_cell={
                                "backgroundColor": "#0f1629",
                                "color": "#e2e8f0",
                                "border": "1px solid #1e3a5f",
                                "padding": "6px 10px",
                                "fontSize": "0.82rem",
                                "textAlign": "left",
                                "whiteSpace": "normal",
                                "height": "auto",
                            },
                            style_header={
                                "backgroundColor": "#1a2540",
                                "color": "#00d4ff",
                                "fontWeight": "600",
                                "border": "1px solid #1e3a5f",
                                "fontSize": "0.82rem",
                            },
                            style_data_conditional=[
                                {"if": {"row_index": "odd"}, "backgroundColor": "#111827"},
                                {"if": {"filter_query": '{label} = "Enthusiastic"'},
                                 "color": "#22c55e"},
                                {"if": {"filter_query": '{label} = "Neutral"'},
                                 "color": "#3b82f6"},
                                {"if": {"filter_query": '{label} = "Critical/Skeptical"'},
                                 "color": "#f97316"},
                                {"if": {"filter_query": '{label} = "Conspiratorial"'},
                                 "color": "#ef4444"},
                                {"if": {"state": "selected"},
                                 "backgroundColor": "#1a2540",
                                 "border": "1px solid #00d4ff"},
                            ],
                        ),
                    ),
                ], width=12, lg=7),

                # Right column: selected tweet + controls
                dbc.Col([
                    dbc.Textarea(
                        id="cmp-tweet-display",
                        value="",
                        rows=6,
                        style={
                            "background": "#111827", "color": "#e2e8f0",
                            "border": "1px solid #1e3a5f", "borderRadius": "6px",
                            "fontSize": "0.9rem", "width": "100%",
                        },
                    ),
                    html.Div(style={"height": "8px"}),
                    html.Div(
                        id="cmp-true-label",
                        style={"color": "#94a3b8", "fontSize": "0.85rem", "marginBottom": "10px"},
                    ),
                    dbc.Row([
                        dbc.Col(
                            dbc.Button(
                                "Tweet", id="cmp-random-btn",
                                color="secondary", className="w-100",
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Classify All", id="cmp-classify-btn",
                                color="primary", className="w-100",
                            ),
                            width=6,
                        ),
                    ]),
                ], width=12, lg=5),
            ], className="mb-3"),
            dcc.Loading(
                html.Div(id="cmp-results"),
                type="circle", color="#00d4ff",
            ),
            dcc.Store(id="cmp-tweet-store"),
        ]),
    ], id="models-inner-tabs", active_tab="models-nn"),
], fluid=True, style={"paddingBottom": "40px"})


# ── LIVE TEST ──────────────────────────────────────────────────────────────────

live_test_content = dbc.Container([
    dbc.Row([
        # ── Left panel: controls ────────────────────────────────────────────
        dbc.Col([
            html.Div("Live Inference", className="section-header"),
            _download_notice(),
            dbc.Alert(
                "Type or paste an Artemis II tweet. The same text cleaning pipeline "
                "used during training is applied before inference.",
                color="info", style={"fontSize": "0.85rem"},
            ),
            dbc.Textarea(
                id="live-tweet-input",
                placeholder="Type or paste an Artemis II tweet here…",
                rows=5,
                style={
                    "background": "#111827", "color": "#e2e8f0",
                    "border": "1px solid #1e3a5f", "borderRadius": "8px",
                    "fontSize": "0.95rem", "width": "100%", "marginBottom": "12px",
                },
            ),
            html.Div("Select models to run:",
                     style={"color": "#cbd5e1", "fontSize": "0.85rem", "marginBottom": "6px"}),
            dbc.Checklist(
                id="live-model-selector",
                options=[
                    {
                        "label": (f"{n}  [available]" if _model_file_ok(n)
                                  else f"{n}  [run download_models.py]"),
                        "value": n,
                    }
                    for n in MODEL_NAMES
                ],
                value=[n for n in MODEL_NAMES if _model_file_ok(n)],
                style={"color": "#e2e8f0", "marginBottom": "14px"},
            ),
            dbc.Button("Analyze", id="live-analyze-btn", color="primary",
                       size="lg", className="w-100"),

            # ── Groq API key ────────────────────────────────────────────────
            html.Hr(style={"borderColor": "#1e3a5f",
                            "marginTop": "22px", "marginBottom": "18px"}),
            html.Div("LLM Explanations", className="section-header"),
            dbc.Input(
                id="live-groq-key",
                type="password",
                placeholder="Paste your Groq API key here…",
                debounce=False,
                style={
                    "background": "#111827", "color": "#e2e8f0",
                    "border": "1px solid #1e3a5f", "borderRadius": "8px",
                    "fontSize": "0.88rem", "marginBottom": "8px",
                },
            ),
            html.P([
                "Paste your Groq API key to enable per-model LLM explanations. ",
                html.A("Get a free key at console.groq.com",
                       href="https://console.groq.com", target="_blank",
                       style={"color": "#00d4ff"}),
                ". The key is used only for the current session and is never "
                "stored or written to disk.",
            ], style={"color": "#64748b", "fontSize": "0.78rem", "lineHeight": "1.55"}),
            (dbc.Alert(
                "groq package not installed; run: pip install groq",
                color="warning", style={"fontSize": "0.78rem", "marginTop": "6px"},
            ) if not GROQ_PKG_OK else html.Div()),
        ], width=12, lg=4),

        # ── Right panel: results (one card per model) ───────────────────────
        dbc.Col([
            html.Div("Results", className="section-header"),
            dcc.Loading(
                html.Div(id="live-results"),
                type="circle", color="#00d4ff",
            ),
        ], width=12, lg=8),
    ]),
    dcc.Store(id="live-prediction-store"),
], fluid=True, style={"paddingBottom": "40px"})


# ══════════════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="ARTEMIS II",
    suppress_callback_exceptions=True,
)
server = app.server

_navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand(
            html.Span([_moon_icon, "ARTEMIS II"]),
            href="#",
        ),
    ], fluid=True),
    color="dark", dark=True, sticky="top",
)

app.layout = html.Div([
    _navbar,
    html.Div(
        [
            html.Div(
                "Università Cattolica del Sacro Cuore",
                style={"fontWeight": "600", "color": "#e2e8f0",
                       "fontSize": "0.72rem", "lineHeight": "1.4",
                       "letterSpacing": "0.01em"},
            ),
            html.Div(
                "Text Mining and Data Visualization",
                style={"color": "#94a3b8", "fontSize": "0.65rem",
                       "letterSpacing": "0.01em"},
            ),
        ],
        style={
            "position": "fixed",
            "top": "62px",
            "right": "14px",
            "zIndex": "100",
            "textAlign": "right",
            "pointerEvents": "none",
            "background": "rgba(8,9,26,0.75)",
            "borderRadius": "4px",
            "padding": "4px 8px",
            "border": "1px solid rgba(30,58,95,0.6)",
        },
    ),
    dbc.Container([
        html.Div(style={"height": "16px"}),
        dbc.Tabs([
            dbc.Tab(home_content,      label="Home",      tab_id="tab-home"),
            dbc.Tab(dataset_content,   label="Dataset",   tab_id="tab-dataset"),
            dbc.Tab(eda_content,       label="EDA",       tab_id="tab-eda"),
            dbc.Tab(models_content,    label="Models",    tab_id="tab-models"),
            dbc.Tab(live_test_content, label="Live Test", tab_id="tab-live"),
        ], id="main-tabs", active_tab="tab-home"),
    ], fluid=True, style={"maxWidth": "1400px"}),
    html.Footer(
        dbc.Container(
            html.P(
                "Dataunderdogs · ARTEMIS II Sentiment Analysis · 2026",
                style={"color": "#94a3b8", "fontSize": "0.78rem", "textAlign": "center",
                       "marginTop": "40px", "paddingBottom": "20px"},
            ),
            fluid=True,
        )
    ),
], style={"background": "#08091a", "minHeight": "100vh"})


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("github-members-modal", "is_open"),
    Input("home-github-btn", "n_clicks"),
    State("github-members-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_github_modal(n_clicks, is_open):
    if n_clicks:
        return True
    return is_open


app.clientside_callback(
    """
    function(n1, n2, n3, n4, n5) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx || !ctx.triggered || !ctx.triggered.length) {
            return [null, null, null, null, null];
        }
        var triggerId = ctx.triggered[0].prop_id.split('.')[0];
        var targetId = 'model-section-' + triggerId.replace('jump-', '');
        var el = document.getElementById(targetId);
        if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
        return [null, null, null, null, null];
    }
    """,
    [Output(f"jump-{m}", "n_clicks") for m in MODEL_NAMES],
    [Input(f"jump-{m}", "n_clicks") for m in MODEL_NAMES],
    prevent_initial_call=True,
)


@app.callback(
    Output("tweet-detail-modal", "is_open"),
    Output("tweet-detail-modal-body", "children"),
    Output("master-table", "active_cell"),
    Input("master-table", "active_cell"),
    Input("tweet-modal-close", "n_clicks"),
    State("master-table", "derived_virtual_data"),
    prevent_initial_call=True,
)
def handle_tweet_modal(active_cell, close_clicks, rows):
    ctx = callback_context
    if not ctx.triggered:
        return False, no_update, no_update
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "tweet-modal-close":
        return False, no_update, no_update
    if active_cell is None or not rows:
        return False, no_update, no_update
    row = rows[active_cell["row"]]
    label    = str(row.get("Sentiment_label", ""))
    source   = str(row.get("source", ""))
    original = str(row.get("text", ""))
    cleaned  = str(row.get("cleaned_text", ""))
    label_color = {
        "Enthusiastic":       "#22c55e",
        "Neutral":            "#3b82f6",
        "Critical/Skeptical": "#f97316",
        "Conspiratorial":     "#ef4444",
    }.get(label, "#e2e8f0")
    def _pre(text):
        return html.Pre(
            text,
            style={
                "color": "#e2e8f0",
                "background": "#111827",
                "border": "1px solid #1e3a5f",
                "borderRadius": "6px",
                "padding": "12px 14px",
                "fontSize": "0.84rem",
                "lineHeight": "1.65",
                "whiteSpace": "pre-wrap",
                "wordBreak": "break-word",
                "margin": 0,
                "minHeight": "60px",
            },
        )
    body = html.Div([
        html.Div(
            html.Span([
                html.Span(label, style={"color": label_color, "fontWeight": "700"}),
                html.Span(f"  -  {source}", style={"color": "#94a3b8"}),
            ]),
            style={"marginBottom": "14px"},
        ),
        dbc.Row([
            dbc.Col([
                html.Div("Original", style={
                    "color": "#00d4ff", "fontWeight": "700", "fontSize": "0.75rem",
                    "textTransform": "uppercase", "letterSpacing": "0.06em",
                    "marginBottom": "5px",
                }),
                _pre(original),
            ], width=12, lg=6),
            dbc.Col([
                html.Div("Cleaned", style={
                    "color": "#22c55e", "fontWeight": "700", "fontSize": "0.75rem",
                    "textTransform": "uppercase", "letterSpacing": "0.06em",
                    "marginBottom": "5px",
                }),
                _pre(cleaned),
            ], width=12, lg=6, style={"marginTop": "12px"}),
        ]),
    ])
    return True, body, None


@app.callback(
    Output("wc-content", "children"),
    Input("eda-inner-tabs", "active_tab"),
)
def render_wordclouds(active_tab):
    if active_tab != "eda-wc":
        return no_update
    if not WC_OK:
        return dbc.Alert(
            "Install the wordcloud library (pip install wordcloud) to enable word clouds.",
            color="warning",
        )
    if df_master is None:
        return dbc.Alert("Dataset not available.", color="warning")

    colormaps = {
        "global":             "plasma",
        "Enthusiastic":       "Greens",
        "Neutral":            "Blues",
        "Conspiratorial":     "Reds",
        "Critical/Skeptical": "Oranges",
    }
    lemma_ok = bool(_lemma_docs) and len(_lemma_docs) == len(df_eda)
    for key, cmap in colormaps.items():
        if key not in _WC_CACHE:
            if lemma_ok:
                if key == "global":
                    # Notebook cell-78: join all lemmas (artemis included), no WC stopwords
                    corpus   = " ".join(_lemma_docs)
                    stops    = set()   # spaCy already removed English stops
                    max_w    = 100
                else:
                    # Notebook cell-82: per-class lemmas, filter exactly 5 domain words
                    cls_idx  = df_eda[df_eda["Sentiment_label"] == key].index.tolist()
                    tokens   = [
                        tok
                        for i in cls_idx if i < len(_lemma_docs)
                        for tok in _lemma_docs[i].split()
                        if tok not in _WC_DOMAIN_STOPS_NB
                    ]
                    corpus   = " ".join(tokens)
                    stops    = set()   # spaCy already removed English stops
                    max_w    = 80
            else:
                # Fallback when spaCy lemma docs are unavailable
                if key == "global":
                    corpus = " ".join(df_eda["cleaned_text"].fillna("").astype(str).tolist())
                else:
                    corpus = " ".join(
                        df_eda[df_eda["Sentiment_label"] == key]["cleaned_text"]
                        .fillna("").astype(str).tolist()
                    )
                stops = None   # use default (ENGLISH_STOP_WORDS | DOMAIN_STOPS)
                max_w = 100 if key == "global" else 80
            _WC_CACHE[key] = _wc_b64(corpus, cmap, stopwords=stops, max_words=max_w)

    def _img_el(key: str, title: str) -> html.Div:
        src = _WC_CACHE.get(key)
        if not src:
            return dbc.Alert(f"Could not generate word cloud for {title}.", color="warning")
        return html.Div([
            html.H6(title, style={"color": "#e2e8f0", "textAlign": "center", "marginBottom": "8px",
                                  "fontWeight": "600"}),
            html.Img(src=src, style={"width": "100%", "borderRadius": "8px"}),
        ])

    return html.Div([
        _img_el("global", "Global Word Cloud: All Classes"),
        html.Div(style={"height": "20px"}),
        html.Div("Per-Class Word Clouds (domain anchors removed)", className="section-header"),
        dbc.Row([
            dbc.Col(_img_el("Enthusiastic",       "Enthusiastic"),       width=6),
            dbc.Col(_img_el("Neutral",             "Neutral"),             width=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(_img_el("Conspiratorial",      "Conspiratorial"),      width=6),
            dbc.Col(_img_el("Critical/Skeptical",  "Critical / Skeptical"), width=6),
        ]),
    ])


@app.callback(
    Output("eda-ling-content", "children"),
    Input("eda-inner-tabs", "active_tab"),
    prevent_initial_call=True,
)
def render_ling_content(active_tab):
    global nlp_cache, _nlp_spacy_computed
    if active_tab != "eda-ling":
        return no_update

    if not nlp_cache:
        if not SPACY_OK:
            return dbc.Alert(
                "spaCy is not installed, run: "
                "pip install spacy && python -m spacy download en_core_web_sm",
                color="warning", style={"fontSize": "0.85rem"},
            )
        with _nlp_lock:
            if not _nlp_spacy_computed:
                nlp_cache = _compute_nlp_cache()
                _nlp_spacy_computed = True

    if not nlp_cache:
        return dbc.Alert(
            "NLP computation produced no results.", color="danger",
            style={"fontSize": "0.85rem"},
        )

    stats = [
        ("Total Sentences",
         f"{nlp_cache.get('total_sentences', 0):,}"
         if isinstance(nlp_cache.get("total_sentences"), int) else "N/A"),
        ("Avg Sentences / Tweet", f"{nlp_cache.get('sents_per_doc_mean', 0):.1f}"),
        ("Avg Tokens / Sentence", f"{nlp_cache.get('sent_length_mean', 0):.1f}"),
    ]
    return html.Div([
        dbc.Alert(
            "NLP analysis powered by spaCy (en_core_web_sm). "
            "Computed on first visit, cached for the session.",
            color="info", style={"fontSize": "0.85rem"},
        ),
        dbc.Row([
            dbc.Col(html.Div([
                html.Div(v, className="stat-value"),
                html.Div(k, className="stat-label"),
            ], className="stat-badge"), width=4)
            for k, v in stats
        ]),
        html.Div(style={"height": "16px"}),
        dcc.Graph(figure=make_pos_fig(),          config={"displayModeBar": False}),
        html.Div(style={"height": "8px"}),
        dcc.Graph(figure=make_ner_type_fig(),     config={"displayModeBar": False}),
        html.Div(style={"height": "8px"}),
        dcc.Graph(figure=make_top_entities_fig(), config={"displayModeBar": False}),
    ])


@app.callback(
    Output("live-results", "children"),
    Output("live-prediction-store", "data"),
    Input("live-analyze-btn", "n_clicks"),
    State("live-tweet-input", "value"),
    State("live-model-selector", "value"),
    State("live-groq-key", "value"),
    prevent_initial_call=True,
)
def run_live_inference(n_clicks, tweet_text, selected_models, groq_key):
    if not tweet_text or not tweet_text.strip():
        return dbc.Alert("Please enter a tweet to analyze.", color="warning"), no_update
    if not selected_models:
        return dbc.Alert("Please select at least one model.", color="warning"), no_update

    cleaned = clean_tweet(tweet_text)
    api_key = (groq_key or "").strip()

    header = html.P(
        [html.B("Cleaned text: "),
         html.Code(cleaned[:200] + ("…" if len(cleaned) > 200 else ""))],
        style={"color": "#94a3b8", "fontSize": "0.82rem", "marginBottom": "16px"},
    )

    # ── Phase 1: assess the tweet ONCE, independent of any model ──────────────
    ref_class: str | None = None
    ref_block = html.Div()

    if api_key and GROQ_PKG_OK:
        ref_result = get_groq_reference(tweet_text, api_key)
        if ref_result is not None:
            rc, rj = ref_result
            if rc.startswith("[Groq error:"):
                ref_block = dbc.Alert(
                    [html.B("LLM Reference Error: "), rc],
                    color="danger",
                    style={"fontSize": "0.85rem", "marginBottom": "16px"},
                )
            else:
                ref_class = rc
                ref_color = CLASS_COLORS.get(rc, "#a78bfa")
                ref_block = html.Div([
                    html.Div([
                        html.Span("LLM Reference Assessment",
                                  style={"color": "#a78bfa", "fontWeight": "700",
                                         "fontSize": "0.75rem",
                                         "textTransform": "uppercase",
                                         "letterSpacing": "0.06em"}),
                        html.Span(f"  ·  Groq / {GROQ_MODEL}",
                                  style={"color": "#475569", "fontSize": "0.72rem"}),
                    ], style={"marginBottom": "8px"}),
                    html.Div([
                        html.Span("LLM reference assessment of the tweet:  ",
                                  style={"color": "#94a3b8", "fontSize": "0.88rem"}),
                        html.Span(rc,
                                  style={"color": ref_color, "fontWeight": "700",
                                         "fontSize": "0.95rem"}),
                        html.Br(),
                        html.Span(rj,
                                  style={"color": "#e2e8f0", "fontSize": "0.88rem",
                                         "lineHeight": "1.7"}),
                    ], style={"marginBottom": "8px"}),
                    html.P(
                        f"Note: this reference judgment is itself an independent LLM "
                        f"opinion (Groq / {GROQ_MODEL}, T={LLM_TEMPERATURE}) and may be "
                        "wrong. It is not a verified ground-truth label!",
                        style={"color": "#475569", "fontSize": "0.74rem",
                               "fontStyle": "italic", "margin": 0},
                    ),
                ], style={
                    "background": "#0d1526",
                    "border": "1px solid #a78bfa40",
                    "borderLeft": "3px solid #a78bfa",
                    "borderRadius": "10px",
                    "padding": "14px 18px",
                    "marginBottom": "20px",
                })

    store_data = {"tweet": tweet_text, "predictions": {}}
    model_blocks: list = [header, ref_block]

    # ── Phase 2: per-model inference + comparison against Phase-1 reference ───
    for model_name in selected_models:
        probs_dict, elapsed = run_inference(tweet_text, model_name)
        if probs_dict:
            store_data["predictions"][model_name] = {
                "probs":      probs_dict,
                "predicted":  max(probs_dict, key=probs_dict.get),
                "elapsed_ms": elapsed,
            }
        predicted = max(probs_dict, key=probs_dict.get) if probs_dict else None
        comparison_text: str | None = None
        if api_key and GROQ_PKG_OK and ref_class and predicted:
            comparison_text = get_groq_comparison(
                tweet_text, ref_class, predicted, api_key
            )
        model_blocks.append(
            _live_model_block(model_name, probs_dict, elapsed,
                              comparison_text, bool(api_key))
        )

    return html.Div(model_blocks), store_data


@app.callback(
    Output("cmp-tweet-display", "value"),
    Output("cmp-true-label", "children"),
    Output("cmp-tweet-store", "data"),
    Input("cmp-tweet-table", "selected_rows"),
    prevent_initial_call=True,
)
def select_tweet_from_table(selected_rows):
    if not selected_rows or _inference_batch.empty:
        return no_update, no_update, no_update
    idx   = selected_rows[0]
    row   = _inference_batch.iloc[idx]
    text  = str(row.get("text", ""))
    label = str(row.get("label", ""))
    label_html = [
        "True label: ",
        html.Span(label, style={"color": CLASS_COLORS.get(label, "#94a3b8"),
                                 "fontWeight": "700"}),
    ]
    return text, label_html, {"tweet": text, "true_label": label}


@app.callback(
    Output("cmp-tweet-table", "selected_rows"),
    Input("cmp-random-btn", "n_clicks"),
    prevent_initial_call=True,
)
def select_random_tweet(n_clicks):
    if _inference_batch.empty:
        return []
    return [random.randint(0, len(_inference_batch) - 1)]


@app.callback(
    Output("cmp-results", "children"),
    Input("cmp-classify-btn", "n_clicks"),
    State("cmp-tweet-store", "data"),
    prevent_initial_call=True,
)
def run_comparison_inference(n_clicks, store_data):
    if not store_data or not store_data.get("tweet"):
        return dbc.Alert(
            "Select a tweet from the list (or press Tweet) before classifying.",
            color="warning",
        )
    tweet = store_data["tweet"]
    cards: list = [
        html.P(
            html.B("Running all 5 models. First call per model loads weights…"),
            style={"color": "#94a3b8", "fontSize": "0.85rem", "marginBottom": "8px"},
        ),
    ]
    for name in MODEL_NAMES:
        probs_dict, elapsed = run_inference(tweet, name)
        cards.append(model_result_card(name, probs_dict, elapsed))
    return html.Div(cards)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("[startup] Dashboard ready. Navigate to http://127.0.0.1:8050")
    app.run(debug=False, host="127.0.0.1", port=8050)
