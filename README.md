# Cerdas

Indonesian e-commerce review classifier.

## What it does

Takes Indonesian product reviews and predicts:
- Product category (5 classes)
- Sentiment (positive / neutral / negative)

Built on the Tokopedia 2019 dataset (40,607 reviews). The plan is to extend this with semantic search over similar past complaints and an LLM layer that drafts reply suggestions, so the system can route incoming complaints to the right CS team and suggest a first-pass response.

## Setup

Standard Python project. Needs Python 3.10+ and the dataset.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/farhan999/tokopedia-product-reviews-2019) and save as `data/raw/reviews.csv`. Then run:

```bash
python src/data_prep.py
```

## Stack

Python, scikit-learn, pandas. Adding HuggingFace transformers and ChromaDB later.

## Why this dataset

Indonesian language, e-commerce domain, public license — closest match to Blibli's actual data.