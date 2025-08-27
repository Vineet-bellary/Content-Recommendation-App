# eda_plots.py
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


# 1. Movies vs TV Shows
def plot_movie_vs_tv(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="type", palette="Set1", width=0.4)
    plt.title("Movies VS TV Shows on Netflix", fontsize=14)
    plt.xlabel("Type of Content", fontsize=12)
    plt.ylabel("Number of Titles", fontsize=12)
    return plt.gcf()


# 2. Titles Added Each Year
def plot_titles_per_year(df):
    content_by_year = df["year_added"].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        x=content_by_year.index, y=content_by_year.values, marker="o", linewidth=2
    )
    plt.title("Number of Titles Added to Netflix Each Year", fontsize=16)
    plt.xlabel("Year Added", fontsize=12)
    plt.ylabel("Number of Titles", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()


# 3. Movies vs TV Shows per Year
def plot_movies_vs_tv_per_year(df):
    grouped_data = df.groupby(["year_added", "type"]).size().unstack()
    plt.figure(figsize=(12, 6))
    grouped_data.plot(kind="line", marker="o", linewidth=2)
    plt.title("Movies vs TV Shows Added Each Year on Netflix", fontsize=16)
    plt.xlabel("Year Added", fontsize=12)
    plt.ylabel("Number of Titles", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()


# 4. Top Countries
def plot_top_countries(df):
    df_country = df.copy()
    df_country["country"] = df_country["country"].str.split(", ")
    df_country = df_country.explode("country")

    country_counts = df_country["country"].value_counts().head(10)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=country_counts.values, y=country_counts.index, palette="Set1")
    plt.title("Top 10 Countries Contributing Netflix Content", fontsize=16)
    plt.xlabel("Number of Titles")
    plt.ylabel("Country")
    plt.tight_layout()
    return plt.gcf()
