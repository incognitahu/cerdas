import pandas as pd

df = pd.read_csv("data/raw/reviews.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("\n--- Shape ---")
print(df.shape)
print("\n--- Columns ---")
print(df.columns.tolist())
print("\n--- Dtypes ---")
print(df.dtypes)
print("\n--- First 3 rows ---")
print(df.head(3))
print("\n--- Null counts ---")
print(df.isnull().sum())

print("\n--- Sample reviews ---")

print("\nShortest reviews:")
pd.set_option('display.max_colwidth', 800)
df['_text_length'] = df['text'].str.len()
print(df.nsmallest(3, '_text_length')[['_text_length', 'text']])

print("\nLongest reviews:")
print(df.nlargest(3, '_text_length')[['_text_length', 'text']])

df.drop(columns=['_text_length'])

