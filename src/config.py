"""
config.py — Central configuration for Cerdas.

LEARN: Why a config file?
    - One place to change settings → changes propagate everywhere
    - No hardcoded paths scattered across scripts
    - Makes your code reproducible on different machines
    - Industry standard pattern

Dataset: Tokopedia Product Reviews 2019
    Source: Kaggle / HuggingFace (farhamu/tokopedia-product-reviews-2019)
    License: MIT
    Size: 40,607 reviews · 5 categories · 3,647 unique products
    Language: Bahasa Indonesia
    Columns: text, rating, category, product_name, product_id, sold, shop_id, product_url
"""

from pathlib import Path

# =============================================================================
# Paths — all relative to PROJECT_ROOT
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RAW_DATA_FILE = RAW_DATA_DIR / "reviews.csv"
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

# =============================================================================
# Dataset schema — Tokopedia 2019
# =============================================================================
TEXT_COLUMN = "text"
RATING_COLUMN = "rating"
LABEL_COLUMN = "category"
# 5 categories: elektronik, fashion, olahraga, handphone, pertukangan

# =============================================================================
# Data split
# =============================================================================
TEST_SIZE = 0.2
RANDOM_STATE = 42  # LEARN: always set seeds for reproducibility

# =============================================================================
# Text filtering
# =============================================================================
MIN_TEXT_LENGTH = 5       # drop reviews shorter than this
MAX_TEXT_LENGTH = 2000    # truncate ridiculously long ones

# =============================================================================
# Classifier hyperparameters (used in Module 1)
# =============================================================================
# TF-IDF + LogReg baseline
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)     # unigrams + bigrams

# IndoBERT fine-tuning (Week 2 stretch goal)
BERT_MODEL_NAME = "indobenchmark/indobert-base-p1"
BERT_MAX_LENGTH = 128
BERT_BATCH_SIZE = 16
BERT_LEARNING_RATE = 2e-5
BERT_EPOCHS = 3

# =============================================================================
# Ensure directories exist
# =============================================================================
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
