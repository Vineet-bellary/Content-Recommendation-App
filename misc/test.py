import pandas as pd

df = pd.read_csv("dataset/final_cleaned_dataset.csv")

genres = df["listed_in"].apply(lambda x: [g.strip() for g in x.split(",")])
genres = genres.explode().reset_index(drop=True).unique()
for i, genre in enumerate(genres):
    print(f"{i+1}: {genre}")
