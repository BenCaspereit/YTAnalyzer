# YTAnalyzer

A Python-based data pipeline designed to scrape YouTube comments at scale, apply language filtering, and perform multi-task NLP classification (Sentiment, Emotion, Intention, Theme) using PyTorch and Hugging Face Transformers.

The core architectural goal is to use heavy, pre-trained models (BERT/DeBERTa) as a weak supervision layer to automatically label a massive raw dataset, bypassing manual data annotation.

## Features

* **API Scraping:** Handles official YouTube Data API v3 pagination dynamically via `list_next` to prevent data loss.
* **Language Validation:** Deterministic filtering using `langdetect` to enforce English-only datasets before inference.
* **Hardware Acceleration:** Full CUDA integration optimized for local GPUs (tested on RTX 3060 Ti) using Mixed Precision (`torch.amp.autocast`) and batching to maintain low VRAM overhead.
* **4-Way Classification Pipeline:**
  * **Sentiment:** 1-5 Star rating using `nlptown/bert-base-multilingual-uncased-sentiment`.
  * **Emotion:** Multi-class emotion tracking via `bhadresh-savani/bert-base-uncased-emotion`.
  * **Intention:** Dialogue intent analysis via `mindpadi/intent_classifier`.
  * **Theme:** Zero-Shot topic assignment (20 candidate labels) via `microsoft/deberta-v3-large`.

## Dataset & Scale

The codebase is structured to handle larger text payloads without memory leaks:
* **Raw Corpus:** `~750 MB` local JSON storage containing **2,340,125 lines** of unlabelled comment data.
* **Developer Reporting:** Integrated CLI tool to monitor language distribution across scraped targets during execution.

## Tech Stack

* **Core:** Python 3.12, PyTorch (v2.5.1+cu121 / v2.6+), Transformers
* **Data Processing:** Pandas, Google API Client, Langdetect, Emoji
* **Hardware:** NVIDIA CUDA 12.1



     [ YouTube API ]
                 │
        ( fetch_comments.py ) ──> Filters Language (en)
                 │
          [ comments.json ] (2.3 Million Records / ~750MB)
                 │
        ( comment_analyzer.py ) ──> Batch Processing & CUDA (AMP)
                 │
     ┌───────────┼───────────┬───────────┐
     ▼           ▼           ▼           ▼
 Sentiment    Emotion    Intention     Theme
     └───────────┼───────────┴───────────┘
                 │
          [ results.json ] (Labelled Bootstrapped Dataset)

  
