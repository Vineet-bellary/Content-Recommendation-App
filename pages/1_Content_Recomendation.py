import os
import pandas as pd
import pickle
import streamlit as st
from scripts.content_recomender import recommend

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# Load dataset
@st.cache_data
def Load_data():
    return pd.read_csv(os.path.join(MODEL_DIR, "recommender_df.csv"))


df = Load_data()

st.title("🎬 Netflix Content Recommendation System")

selected_title = st.selectbox(
    "Select a Movie/Show", [None] + sorted(df["title"].unique())
)

if st.button("Get Recommendations"):
    if selected_title:
        recs = recommend(selected_title)
        # recs = set(recs)
        # recs = list(recs)
        if recs:
            st.subheader("🔍 Top 10 Recommendations")
            for i, r in enumerate(recs, start=1):
                st.markdown(
                    f"""
                    {i}. **{r['title']}**    
                    {r['description']}
                    ---
                """
                )
        else:
            st.info("No recommendations found.")
    else:
        # st.warning("Please select a title to get recommendations.")
        # If no title is selected, show 10 random recommendations
        st.subheader(
            "No specific title was selected, so here are some random suggestions!"
        )
        st.subheader("10 Random Recommendations")
        # Ensure we don't try to get more random samples than available
        num_random_recs = min(10, len(df))
        random_recs_df = df.sample(
            n=num_random_recs, random_state=None
        )  # random_state=None for truly random each time

        for i, row in enumerate(random_recs_df.itertuples(), start=1):
            st.markdown(
                f"""
                {i}. **{row.title}**
                {row.description}
                ---
            """
            )
