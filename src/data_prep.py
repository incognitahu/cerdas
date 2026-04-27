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

df = pd.read_csv("data/raw/reviews.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# print("\n--- Shape ---")
# print(df.shape)
# print("\n--- Columns ---")
# print(df.columns.tolist())
# print("\n--- Dtypes ---")
# print(df.dtypes)
# print("\n--- First 3 rows ---")
# print(df.head(3))
# print("\n--- Null counts ---")
# print(df.isnull().sum())

# print("\n--- Sample reviews ---")

# print("\nShortest reviews:")
# pd.set_option('display.max_colwidth', 800)
# df['_text_length'] = df['text'].str.len()
# print(df.nsmallest(3, '_text_length')[['_text_length', 'text']])

# print("\nLongest reviews:")
# print(df.nlargest(3, '_text_length')[['_text_length', 'text']])

# df.drop(columns=['_text_length'])

# ----------------------------------------------------

# print(repr(clean_text("Hello <b>WORLD</b>! Visit http://example.com   ")))
# print(repr(clean_text("👍👍👍")))
# print(repr(clean_text(None)))
# print(repr(clean_text(123)))
# print(repr(clean_text("Barang BAGUS, packaging RAPI!!!")))


# apply clean_text function to the data frame
df['text_cleaned'] = df['text'].apply(clean_text)
df['text_tfidf'] = df['text'].apply(clean_text_tfidf)

print(df[['text', 'text_cleaned', 'text_tfidf']].head(5))
print(f"\nNull counts:")
print(f"  text_cleaned: {df['text_cleaned'].isnull().sum()}")
print(f"  text_tfidf:   {df['text_tfidf'].isnull().sum()}")
print(f"\nEmpty string counts:")
print(f"  text_cleaned: {(df['text_cleaned'] == '').sum()}")
print(f"  text_tfidf:   {(df['text_tfidf'] == '').sum()}")
