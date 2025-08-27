import streamlit as st
import os
import pickle

# -------------------- Paths -------------------- #
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# -------------------- Load Artifacts -------------------- #
@st.cache_resource  # cache so they dont load every time
def load_artifacts():
    with open(os.path.join(MODELS_DIR, "genre_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "mlb.pkl"), "rb") as f:
        mlb = pickle.load(f)
    return model, vectorizer, mlb


model, vectorizer, mlb = load_artifacts()

# -------------------- Streamlit UI -------------------- #
st.title("🎬 Netflix Genre Predictor")

st.write("Enter a movie/show description and I’ll predict possible genres.")

# User input
description = st.text_area("Movie/Show Description", height=150)

if st.button("Predict Genres"):
    if description.strip():
        # Preprocess text
        import re

        text = description.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        text = text.strip()

        # Transform input
        X_input = vectorizer.transform([text])

        # Predict
        y_pred = model.predict(X_input)
        genres = mlb.inverse_transform(y_pred)

        if genres and genres[0]:
            st.success(f"Predicted Genres: {', '.join(genres[0])}")
        else:
            st.warning("No genres predicted. Try a more descriptive input.")
    else:
        st.error("Please enter a description before predicting.")
