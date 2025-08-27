import os
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "final_cleaned_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------------------
# 1. TEXT CLEANING & TAG CREATION
# -------------------------------
def _norm(s: str) -> str:
    """Lowercase, remove punctuation, collapse spaces. Drop 'unknown'/'nan'."""
    s = str(s).strip().lower()
    if s in ("", "unknown", "nan", "none"):
        return ""
    s = re.sub(r"[^a-z0-9\s]", " ", s)  # keep alphanumerics and spaces only
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_tags(df: pd.DataFrame, weighted: bool = True) -> pd.Series:
    """Create a combined 'tags' column from title, director, cast, genres, and description."""
    required = ["title", "director", "cast", "listed_in", "description"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    title = df["title"].map(_norm)
    director = df["director"].map(_norm)
    cast = df["cast"].map(_norm)
    listed_in = df["listed_in"].map(_norm)
    description = df["description"].map(_norm)

    if weighted:
        tags = (
            (
                title
                + " "
                + director
                + " "
                + cast
                + " "
                + (listed_in + " ") * 2  # double weight for genres
                + (description + " ") * 2  # double weight for description
            )
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    else:
        tags = (
            (title + " " + director + " " + cast + " " + listed_in + " " + description)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    return tags


# -------------------------------
# 2. BUILD RECOMMENDER MODEL
# -------------------------------
def build_recommender(weighted=True):
    """Build TF-IDF vectorizer, cosine similarity matrix, and save artifacts."""
    df = pd.read_csv(DATA_PATH)

    # Create tags if not already present
    if "tags" not in df.columns:
        df["tags"] = make_tags(df, weighted=weighted)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    tag_vectors = vectorizer.fit_transform(df["tags"])

    # Cosine similarity matrix
    similarity = cosine_similarity(tag_vectors)

    # Save artifacts
    with open(os.path.join(MODEL_DIR, "recommender_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    with open(os.path.join(MODEL_DIR, "similarity_matrix.pkl"), "wb") as f:
        pickle.dump(similarity, f)

    df.to_csv(os.path.join(MODEL_DIR, "recommender_df.csv"), index=False)
    print("Recommender model built and artifacts saved successfully!")


# -------------------------------
# 3. GET RECOMMENDATIONS
# -------------------------------
def recommend(title: str, top_n: int = 10):
    """Recommend top N similar titles without duplicates."""
    df = pd.read_csv(os.path.join(MODEL_DIR, "recommender_df.csv"))
    with open(os.path.join(MODEL_DIR, "similarity_matrix.pkl"), "rb") as f:
        similarity = pickle.load(f)

    if title not in df["title"].values:
        return []

    idx = df[df["title"] == title].index[0]
    distances = similarity[idx]
    recommendations = sorted(
        list(enumerate(distances)), key=lambda x: x[1], reverse=True
    )[1:]  # skip itself, take all others first

    results = []
    seen_titles = set()

    for i, score in recommendations:
        rec_title = df.iloc[i]["title"]
        if rec_title not in seen_titles:
            results.append(
                {
                    "title": rec_title,
                    "description": df.iloc[i]["description"],
                }
            )
            seen_titles.add(rec_title)

        if len(results) >= top_n:  # stop when we hit top_n unique
            break

    return results


# -------------------------------
# 4. MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    # STEP 1: Build recommender model
    build_recommender(weighted=True)

    # # STEP 2: Test recommendation
    # test_title = "Breaking Bad"
    # print(f"\n🎬 Top Recommendations for '{test_title}':")
    # print(recommend(test_title))
