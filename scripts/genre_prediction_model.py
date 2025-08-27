import pandas as pd
import numpy as np
import re
import os
import pickle
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score

# -------------------- Paths -------------------- #
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "final_cleaned_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure models/ exists
os.makedirs(MODELS_DIR, exist_ok=True)


# -------------------- Helpers -------------------- #
def clean_text(text: str) -> str:
    """Lowercase and remove non-alphabet characters."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()


# -------------------- Training -------------------- #
def train_and_evaluate_model():
    """Load dataset, preprocess, train ML model, and evaluate."""
    print("Starting training process...")

    # 1. Load and preprocess dataset
    df = pd.read_csv(DATA_PATH)
    df["description"] = df["description"].fillna("").apply(clean_text)

    # Use a safe lambda function for converting genres
    df["genres"] = df["listed_in"].apply(lambda x: [g.strip() for g in x.split(",")])

    # 2. Features (X) and labels (y)
    X = df["description"]
    y = df["genres"]

    # 3. Vectorize text
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_vectorized = vectorizer.fit_transform(X)

    # 4. Encode genres
    mlb = MultiLabelBinarizer()
    y_encoded = mlb.fit_transform(y)

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y_encoded, test_size=0.2, random_state=42
    )

    # ----------------------------------------------------------------------------------------------------------------
    # 6. Train model with GridSearchCV to find the best parameters
    # The model needs to be wrapped for a GridSearch
    base_model = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced")
    )

    # Use a grid search to find the best value for C
    param_grid = {"estimator__C": [0.1, 1.0, 10.0]}

    grid_search = GridSearchCV(
        estimator=base_model, param_grid=param_grid, scoring="f1_micro", cv=3, n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Get the best model and score from the grid search
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # 7. Evaluate the best model on the unseen TEST set
    y_pred = best_model.predict(X_test)
    score = f1_score(y_test, y_pred, average="micro")

    print(f"Best F1 Score on the TEST set: {score:.4f}")
    print(f"Best parameters found: {best_params}")

    return best_model, vectorizer, mlb, df


# -------------------- Save / Load -------------------- #
def save_artifacts(model, vectorizer, mlb, df):
    """Save trained model and preprocessing objects to disk."""
    print("Saving artifacts...")

    with open(os.path.join(MODELS_DIR, "genre_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODELS_DIR, "mlb.pkl"), "wb") as f:
        pickle.dump(mlb, f)

    # Print summary stats
    print("\nArtifacts saved in models/")
    print(f"Total Samples: {len(df)}")
    print(f"Unique Genres: {len(mlb.classes_)}")


# def load_artifacts():
#     """Load trained artifacts from disk for inference."""
#     print("Loading artifacts from models/ ...")

#     with open(os.path.join(MODELS_DIR, "genre_model.pkl"), "rb") as f:
#         model = pickle.load(f)
#     with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "rb") as f:
#         vectorizer = pickle.load(f)
#     with open(os.path.join(MODELS_DIR, "mlb.pkl"), "rb") as f:
#         mlb = pickle.load(f)

#     print("Artifacts loaded successfully!")
#     return model, vectorizer, mlb


# -------------------- Entry Point -------------------- #
if __name__ == "__main__":
    trained_model, trained_vectorizer, trained_mlb, df = train_and_evaluate_model()
    save_artifacts(trained_model, trained_vectorizer, trained_mlb, df)
