import pandas as pd
import re # regex

def clean_text(text: str) -> str:
    """Clean a review text. Returns cleaned string"""
    if not isinstance(text, str): # if text is not a string then nothing (removing not string review)
        return ""
    
    text = re.sub(r"<[^>]+>", " ", text) # HTML tag removal
    text = re.sub(r"http\S+", " ", text) # URL removal 
    text = re.sub(r"\s+", " ", text) # convert multiple spaces into a single space
    text = text.strip() # removing leading and trailing space 
    text = text.lower() # lowercasing

    return text

def clean_text_tfidf(text: str) -> str:
    """Adds elongation collapse on top of clean_text()"""
    text = clean_text(text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    return text

def count_letter_words(text: str) -> int: # counting real worlds
    """Count words that contain at least one letter"""
    if not isinstance(text, str):
        return 0

    words = text.split()
    real_words = [w for w in words if any(c.isalpha() for c in w)]

    return len(real_words)

df = pd.read_csv("data/raw/reviews.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df['text_cleaned'] = df['text'].apply(clean_text)
df['text_tfidf'] = df['text'].apply(clean_text_tfidf)

# df = df[df['text_cleaned'].str.len() >= 5] # Boolean indexing



# Filter useless reviews (emoji-only, numbers-only, single-symbol)
# before = len(df)
df = df[df['text_cleaned'].apply(count_letter_words) >= 2].reset_index(drop=True)
# after = len(df)
# print(f"Filtered {before - after:,} useless reviews ({100*(before-after)/before:.1f}%)")
# print(f"{after:,} reviews remaining")



