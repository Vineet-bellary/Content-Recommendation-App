# pages/1_EDA_Dashboard.py
import streamlit as st
import pandas as pd
from scripts import eda_plots  # Import your plotting functions


# Load dataset - use st.cache_data for efficiency across reruns and pages
@st.cache_data
def load_data():
    return pd.read_csv("dataset/final_cleaned_dataset.csv")


df = load_data()

st.title("📊 Netflix EDA Dashboard")
st.write(
    "Explore various aspects of the Netflix dataset through interactive visualizations."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Movies vs TV", "Movies vs TV per Year", "Yearly Trend", "Countries"]
)

with tab1:
    st.header("Movies vs TV Shows Distribution")
    st.pyplot(eda_plots.plot_movie_vs_tv(df))

with tab2:
    st.header("Movies vs TV Shows Production Per Year")
    st.pyplot(eda_plots.plot_movies_vs_tv_per_year(df))

with tab3:
    st.header("Total Titles Released Per Year")
    st.pyplot(eda_plots.plot_titles_per_year(df))

with tab4:
    st.header("Top Producing Countries")
    st.pyplot(eda_plots.plot_top_countries(df))
# st.write("EDA Dashboard")
