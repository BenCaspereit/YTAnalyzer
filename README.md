# 🤖 YTAnalyzer

> **A scalable multi-task NLP pipeline for collecting, filtering, and automatically annotating millions of YouTube comments using PyTorch and Hugging Face Transformers.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=for-the-badge\&logo=huggingface\&logoColor=black)](https://huggingface.co/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge\&logo=nvidia\&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

---

## 🚀 Overview

**YTAnalyzer** is a Python-based data pipeline designed to process **millions of YouTube comments** and automatically enrich them with NLP annotations.

The project uses a **weak-supervision approach**: instead of manually labeling millions of comments, pre-trained NLP models are used to generate labels for a large raw corpus.

The pipeline performs four independent NLP tasks:

* ⭐ **Sentiment** — 1–5 star sentiment classification
* ❤️ **Emotion** — emotion classification
* 💬 **Intention** — dialogue intent classification
* 🏷️ **Theme** — zero-shot topic classification

The system is optimized for **consumer-grade GPU hardware** and was tested on an **NVIDIA RTX 3060 with 8 GB VRAM**.

---

## 📊 Dataset & Scale

The pipeline was designed with large text datasets in mind.

| Metric           |                      Value |
| ---------------- | -------------------------: |
| 💬 Comments      |              **2,340,125** |
| 💾 Raw JSON data |                **~750 MB** |
| 🌎 Language      |                **English** |
| 🧠 NLP tasks     |                      **4** |
| 🎮 Tested GPU    |   **RTX 3060 Ti/ 8 GB VRAM** |
| ⚡ Benchmark      | **1,000 comments in 28 s** |

The raw corpus contains more than **2.3 million unlabelled YouTube comments** stored locally as JSON.

The pipeline is designed to process this dataset without loading the complete corpus into GPU memory at once.

---

## 🔍 Data Collection

Comments are retrieved through the official **YouTube Data API v3**.

Pagination is handled dynamically using `list_next`, ensuring that subsequent result pages are processed correctly rather than relying on a fixed number of API requests.

This allows the collector to handle targets with large comment volumes while avoiding incomplete datasets caused by prematurely stopping pagination.

---

## 🌍 Language Filtering

Before expensive model inference, comments are filtered using [`langdetect`](https://pypi.org/project/langdetect/).

Only comments detected as **English** are passed to the NLP classification stage.

This deterministic preprocessing step reduces unnecessary inference and keeps the resulting dataset consistent with the intended language scope.

The project also includes a CLI reporting mechanism to monitor the detected language distribution of scraped targets during execution.

---

## 🤖 Multi-Task NLP Classification

The core of YTAnalyzer is the automated annotation stage.

Instead of manually labeling millions of comments, specialized pre-trained models are used to generate labels.

### ⭐ Sentiment

**Model:** `nlptown/bert-base-multilingual-uncased-sentiment`

Produces a **1–5 star sentiment rating** for each comment.

```text
Comment
   │
   ▼
BERT Sentiment Model
   │
   ▼
⭐ 1  2  3  4  5
```

### ❤️ Emotion

**Model:** `bhadresh-savani/bert-base-uncased-emotion`

Classifies comments according to their emotional content.

### 💬 Intention

**Model:** `mindpadi/intent_classifier`

Analyzes the underlying **dialogue intent** of comments.

### 🏷️ Theme

**Model:** `valhalla/distilbart-mnli-12-1`

A zero-shot classification model assigns one of **20 candidate themes** to each comment.

This makes it possible to introduce topic categories without training a dedicated classifier from scratch.

---

## ⚡ GPU & Performance Optimization

Running multiple Transformer models over millions of comments is computationally expensive.

YTAnalyzer therefore focuses heavily on efficient inference.

### Mixed Precision

Model inference uses PyTorch's:

```python
torch.amp.autocast
```

Mixed-precision inference reduces memory consumption and allows the GPU to process larger batches more efficiently.

### Batching

Comments are processed in batches instead of individually:

```text
Individual inference:

Comment → Model
Comment → Model
Comment → Model
...

Batched inference:

┌─────────┐
│ Comment │
│ Comment │ ──► Model ──► Predictions
│ Comment │
│ Comment │
└─────────┘
```

This significantly improves GPU utilization while keeping VRAM usage within the limits of consumer hardware.

### Benchmark

On an **RTX 3060 Ti with 8 GB VRAM**, a test run of **1,000 comments completed in approximately 28 seconds**.

This corresponds to roughly:

> **35.7 comments / second**

The benchmark demonstrates that large-scale Transformer inference can be performed locally without requiring enterprise-grade GPU hardware.

---

## 🏗️ Design Goals

The project was built around several engineering goals:

### 📈 Scalability

The pipeline is designed to work with datasets containing **millions of text records**, rather than only small experimental datasets.

### 🧠 Weak Supervision

Pre-trained models act as automatic labeling systems, making it possible to create a large annotated dataset without manually labeling every comment.

### ⚡ Efficient Inference

Batching and mixed precision reduce the computational and memory requirements of Transformer inference.

### 🧩 Modular Processing

Data collection, language filtering, and NLP classification are separated into distinct processing stages.

This makes individual components easier to test, optimize, and replace.

---

## 🛠️ Tech Stack

| Technology                    | Purpose                            |
| ----------------------------- | ---------------------------------- |
| **Python 3.12**               | Core pipeline                      |
| **PyTorch**                   | Model inference & GPU acceleration |
| **Hugging Face Transformers** | Pre-trained NLP models             |
| **Pandas**                    | Data processing                    |
| **YouTube Data API v3**       | Comment collection                 |
| **Langdetect**                | Language detection                 |
| **Emoji**                     | Emoji-aware text processing        |
| **NVIDIA CUDA 12.1**          | GPU acceleration                   |

---

## 📦 Models

| Task         | Model                                              | Approach                   |
| ------------ | -------------------------------------------------- | -------------------------- |
| ⭐ Sentiment  | `nlptown/bert-base-multilingual-uncased-sentiment` | 1–5 star classification    |
| ❤️ Emotion   | `bhadresh-savani/bert-base-uncased-emotion`        | Multi-class classification |
| 💬 Intention | `mindpadi/intent_classifier`                       | Intent classification      |
| 🏷️ Theme    | `valhalla/distilbart-mnli-12-1`                    | Zero-shot classification   |

