# Cerdas — AI-Powered E-commerce Review Intelligence

> End-to-end AI system for Indonesian e-commerce complaint analysis: classical ML classification, semantic retrieval (RAG), LLM-generated responses, and smart routing.

**Target:** Blibli Data Scientist Intern 2026 · **Stack:** Python · scikit-learn · Hugging Face · ChromaDB · OpenAI

---

## Project Status

**Current:** Week 1 — data preparation complete. EDA in progress.

---

## Dataset

**Tokopedia Product Reviews 2019** — 40,607 Indonesian-language product reviews across 5 categories.

- Source: https://www.kaggle.com/datasets/farhan999/tokopedia-product-reviews?resource=download 
- License: MIT
- Collection period: 2019
- Categories: elektronik, fashion, olahraga, handphone, pertukangan

### Key data characteristics

| Aspect | Value | Note |
|---|---|---|
| Total reviews | 40,607 | 100% usable — 0 nulls in text |
| Unique products | 3,647 | Good variety |
| Language | Bahasa Indonesia | Match target market |
| Sentiment skew | **93% positive** | Heavy imbalance — affects modeling |
| Dominant category | elektronik (39%) | Minority: pertukangan (5%) |

---

## Why this project (business framing)

Indonesian e-commerce platforms receive thousands of reviews daily. Current customer service teams face three problems:

1. **Volume:** Manual review triage doesn't scale
2. **Latency:** Complaints wait hours before CS sees them
3. **Routing:** Generic CS teams lack category-specific expertise

**Cerdas** addresses this by automatically classifying reviews by category and sentiment, retrieving similar historical cases, generating context-aware reply drafts, and routing complex complaints to the right team.

---

## Architecture

```
  Customer Review (Indonesian)
           │
  ┌────────▼─────────────────┐
  │ Module 1: Classification │
  │  LogReg + TF-IDF (base)  │
  │  IndoBERT (fine-tuned)   │
  │  → category + sentiment  │
  └────────┬─────────────────┘
           │
  ┌────────▼─────────────────┐
  │ Module 2: Retrieval      │
  │  sentence-transformers   │
  │  ChromaDB vector store   │
  │  → top 3 similar cases   │
  └────────┬─────────────────┘
           │
  ┌────────▼─────────────────┐
  │ Module 3: LLM Layer      │
  │  OpenAI gpt-4o-mini      │
  │  → summarize complaint   │
  │  → suggest reply (RAG)   │
  │  → extract aspects       │
  └────────┬─────────────────┘
           │
  ┌────────▼─────────────────┐
  │ Module 4: Router         │
  │  Auto-reply / Escalate / │
  │  Route to category team  │
  └────────┬─────────────────┘
           │
  ┌────────▼─────────────────┐
  │ Module 5: Evaluation     │
  │  Classifier: macro-F1    │
  │  LLM: manual rating 1-5  │
  └──────────────────────────┘
```

---

## Setup

```bash
# 1. Clone & enter
git clone <your-repo-url> cerdas
cd cerdas

# 2. Virtual environment
python -m venv venv
source venv/bin/activate     # Mac/Linux
# venv\Scripts\activate      # Windows

# 3. Install
pip install -r requirements.txt

# 4. Get dataset
# Download from: https://huggingface.co/datasets/farhamu/tokopedia-product-reviews-2019
# Save as: data/raw/reviews.csv

# 5. Run data prep
python src/data_prep.py
```

Expected output: `data/processed/train.csv` (32,483 rows) and `test.csv` (8,121 rows).

---

## Project Structure

```
cerdas/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/            # reviews.csv (gitignored — 10 MB)
│   └── processed/      # train.csv, test.csv (gitignored)
│
├── src/
│   ├── config.py       # paths, column names, hyperparams
│   ├── data_prep.py    # Module 0: clean + split
│   ├── classifier.py   # Module 1: LogReg + IndoBERT
│   ├── retrieval.py    # Module 2: embeddings + ChromaDB
│   ├── llm_layer.py    # Module 3: OpenAI integration
│   ├── router.py       # Module 4: rule-based routing
│   └── evaluation.py   # Module 5: eval framework
│
├── notebooks/
│   └── 01_eda.py       # Exploratory analysis
│
├── eval_data/
│   └── eval_set.json   # 15-20 labeled test cases
│
├── models/             # Trained models (gitignored)
└── chroma_db/          # Vector store (gitignored)
```

---

## Roadmap (4 weeks, 3 hrs/day)

### Week 1 — Data & baseline
- [x] Day 1-2: Project setup, config, data prep
- [ ] Day 3: EDA notebook (`notebooks/01_eda.py`)
- [ ] Day 4-5: TF-IDF + LogReg baseline classifier
- [ ] Day 6-7: Evaluation framework, baseline results

### Week 2 — Advanced models + retrieval
- [ ] Day 1-3: IndoBERT fine-tuning (stretch goal)
- [ ] Day 4-5: Sentence embeddings + ChromaDB setup
- [ ] Day 6-7: Retrieval function + tests

### Week 3 — LLM layer
- [ ] Day 1-2: OpenAI API, prompt engineering basics
- [ ] Day 3-4: Summarize + reply generation + aspect extraction
- [ ] Day 5-6: Rule-based router
- [ ] Day 7: End-to-end integration

### Week 4 — Polish & apply
- [ ] Day 1-2: Evaluation framework, 15 test cases
- [ ] Day 3-4: Streamlit demo UI
- [ ] Day 5: Deploy to Streamlit Cloud
- [ ] Day 6: README polish, architecture diagram, PDF write-up
- [ ] Day 7: Demo video, LinkedIn post, **apply to Blibli + others**

---

## Blibli JD coverage

| # | JD Requirement | Covered by |
|---|---|---|
| 3 | ML + LLMs + GenAI principles | Modules 1, 3 |
| 4 | Python for ML | All modules |
| 5 | Preprocessing, feature engineering, eval | data_prep.py + Module 5 |
| 6 | Prompt engineering + LLM output eval | Module 3 + Module 5 |
| 7 | LLM API integration (OpenAI) | Module 3 |
| 8 | RAG + fine-tuning | Modules 1 (IndoBERT) + 2 (RAG) |
| 9 | Analytical & problem-solving | Business framing + EDA |
| 10 | ML/NLP/LLM project experience | This project |

---

## License

MIT

## Author

[Your name] — built for Blibli Data Scientist Internship application, 2026.
