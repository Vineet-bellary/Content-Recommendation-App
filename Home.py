# app.py
import streamlit as st
import pandas as pd

# ----------------Streamlit App Main Config----------------
# This must be the first Streamlit command and only in the main app.py
st.set_page_config(
    page_title="Content Recommendation System",  # A more generic title for the whole app
    page_icon=":guardsman:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🏡 Welcome to the Content Recommendation App!")
st.write("Navigate to different sections using the sidebar on the left.")
st.write("---")  # Visual separator

st.markdown("### How to use this app:")
st.markdown(
    "- **EDA Dashboard**: Explore insights and visualizations from the dataset."
)
st.markdown("- **Content Recommendation**: Get personalized content suggestions.")
st.markdown("- **Genre Prediction**: Predict the genre of new content.")

# Streamlit will automatically generate sidebar navigation from files in the 'pages' directory.
# You no longer need manual sidebar buttons or st.session_state.page logic here.
