"""
data_prep.py — Module 0: Load and clean raw Tokopedia dataset.

Pipeline:
    1. Load raw CSV (40,607 reviews from Tokopedia 2019)
    2. Drop spurious 'Unnamed: 0' column from CSV export
    3. Clean text (light for BERT, aggressive for TF-IDF)
    4. Filter too-short reviews
    5. Engineer text features (length, caps ratio, etc.)
    6. Derive sentiment from rating (weak supervision)
    7. Stratified train/test split
    8. Save to data/processed/

LEARN (for interview):
    - Always EDA before modeling (see notebooks/01_eda.py)
    - Clean data > fancy model on dirty data
    - Weak supervision from rating is a common industry pattern
    - Stratified split matters when classes are imbalanced

Run: `python src/data_prep.py` from project root.
"""

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Support running as script or module
try:
    from src.config import (
        RAW_DATA_FILE, PROCESSED_TRAIN_FILE, PROCESSED_TEST_FILE,
        TEXT_COLUMN, LABEL_COLUMN, RATING_COLUMN,
        TEST_SIZE, RANDOM_STATE, MIN_TEXT_LENGTH,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.config import (
        RAW_DATA_FILE, PROCESSED_TRAIN_FILE, PROCESSED_TEST_FILE,
        TEXT_COLUMN, LABEL_COLUMN, RATING_COLUMN,
        TEST_SIZE, RANDOM_STATE, MIN_TEXT_LENGTH,
    )


# =============================================================================
# Text cleaning
# =============================================================================
def clean_text(text: str) -> str:
    """
    Light cleaning — preserves structure for BERT.

    LEARN: Don't remove stopwords or stem for BERT.
           BERT models learn these patterns themselves.
           Aggressive cleaning HURTS transformer performance.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r"<[^>]+>", " ", text)          # HTML tags
    text = re.sub(r"http[s]?://\S+", " ", text)   # URLs
    text = re.sub(r"\s+", " ", text)              # collapse whitespace
    return text.strip().lower()


def clean_text_for_tfidf(text: str) -> str:
    """
    Aggressive cleaning for TF-IDF / LogReg.

    LEARN: Classical ML benefits from removing noise (punctuation, digits).
           Two cleaners for two model families = real production pattern.
    """
    text = clean_text(text)
    text = re.sub(r"[^a-z\s]", " ", text)  # keep letters + spaces only
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =============================================================================
# Feature engineering
# =============================================================================
def engineer_features(df: pd.DataFrame, text_col: str = TEXT_COLUMN) -> pd.DataFrame:
    """
    Add derived features from review text.

    LEARN: Feature engineering separates "tutorial followers" from real data
           scientists. These simple features often carry signal the model
           wouldn't extract from text alone (e.g., caps_ratio = frustration).
    """
    df = df.copy()
    df["text_length"] = df[text_col].str.len()
    df["word_count"] = df[text_col].str.split().str.len().fillna(0).astype(int)
    df["avg_word_length"] = df["text_length"] / df["word_count"].replace(0, 1)
    df["caps_ratio"] = df[text_col].apply(
        lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1)
    )
    df["exclaim_count"] = df[text_col].str.count("!")
    df["question_count"] = df[text_col].str.count(r"\?")
    return df


# =============================================================================
# Sentiment from rating (weak supervision)
# =============================================================================
def rating_to_sentiment(rating) -> str:
    """
    Derive sentiment label from star rating.

    LEARN: "Weak supervision" — derive labels from a proxy when true labels
           are expensive. Common in industry ML. Our proxy (rating) is
           stronger than most because users explicitly rate their satisfaction.
    """
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return "neutral"
    if r >= 4:
        return "positive"
    elif r <= 2:
        return "negative"
    else:
        return "neutral"


# =============================================================================
# Pipeline steps
# =============================================================================
def load_raw_data(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load raw CSV. Drops 'Unnamed: 0' if present (common CSV artifact)."""
    if not path.exists():
        raise FileNotFoundError(
            f"\n\nDataset not found at {path}\n"
            f"Download from Kaggle/HuggingFace:\n"
            f"  farhamu/tokopedia-product-reviews-2019\n"
            f"Place at: {path}\n"
        )
    df = pd.read_csv(path)
    # CSV export artifact — useless index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    print(f"Loaded {len(df):,} rows from {path.name}")
    print(f"Columns: {list(df.columns)}")
    return df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Full preparation pipeline."""
    print(f"\nStarting with {len(df):,} rows")

    # 1. Drop null text
    df = df.dropna(subset=[TEXT_COLUMN]).reset_index(drop=True)
    print(f"After dropping null text: {len(df):,} rows")

    # 2. Keep original for display, create cleaned versions
    df["text_raw"] = df[TEXT_COLUMN].astype(str)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).apply(clean_text)
    df["text_tfidf"] = df["text_raw"].apply(clean_text_for_tfidf)

    # 3. Filter too-short
    df = df[df[TEXT_COLUMN].str.len() >= MIN_TEXT_LENGTH].reset_index(drop=True)
    print(f"After filtering short text (<{MIN_TEXT_LENGTH} chars): {len(df):,} rows")

    # 4. Feature engineering (use raw text for caps/punctuation signals)
    df = engineer_features(df, text_col="text_raw")

    # 5. Derive sentiment
    if RATING_COLUMN in df.columns:
        df["sentiment"] = df[RATING_COLUMN].apply(rating_to_sentiment)
        print(f"\nSentiment distribution:")
        print(df["sentiment"].value_counts())
        print(f"\nSentiment proportions (%):")
        print((df["sentiment"].value_counts(normalize=True) * 100).round(1))

    # 6. Category distribution
    if LABEL_COLUMN in df.columns:
        print(f"\nCategory distribution ({LABEL_COLUMN}):")
        print(df[LABEL_COLUMN].value_counts())

    return df


def split_and_save(df: pd.DataFrame) -> None:
    """Stratified train/test split → save CSVs."""
    # LEARN: stratify keeps class proportions balanced across train/test.
    # Critical for imbalanced data (our category "pertukangan" is only 5%).
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COLUMN] if df[LABEL_COLUMN].nunique() > 1 else None,
    )
    train_df.to_csv(PROCESSED_TRAIN_FILE, index=False)
    test_df.to_csv(PROCESSED_TEST_FILE, index=False)

    print(f"\n✓ Train saved: {PROCESSED_TRAIN_FILE.name} ({len(train_df):,} rows)")
    print(f"✓ Test saved:  {PROCESSED_TEST_FILE.name} ({len(test_df):,} rows)")
    print(f"\nTrain category distribution:")
    print(train_df[LABEL_COLUMN].value_counts())


def main():
    df_raw = load_raw_data()
    df_clean = prepare_dataset(df_raw)
    split_and_save(df_clean)
    print("\n✓ Data prep complete.")
    print("  Next step: run `python notebooks/01_eda.py` for exploratory analysis.")


if __name__ == "__main__":
    main()
