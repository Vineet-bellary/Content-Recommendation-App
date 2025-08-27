# Netflix Content Discovery App

This project is a Streamlit-based web application designed to help users explore and discover content on Netflix. It features an Exploratory Data Analysis (EDA) dashboard, a machine learning model for predicting content genres based on descriptions, and a content-based recommendation system.

---

## Features

- **EDA Dashboard**: Dive into interactive visualizations that explore various aspects of the Netflix dataset, including content type distribution, yearly trends, and top contributing countries.

- **Genre Prediction**: Input a movie or TV show description and get an instant prediction of its likely genres. This feature is powered by a fine-tuned Logistic Regression model.

- **Content Recommendation**: Select a movie or TV show title and receive a list of similar content recommendations based on shared characteristics like cast, director, genres, and description.

---

## Technical Stack

- **Language**: Python 3.8+

- **Web Framework**: Streamlit

- **Data Manipulation**: Pandas, NumPy

- **Machine Learning**: Scikit-learn

- **Text Processing**: `re`, `nltk` (implicitly via `stop_words` in `TfidfVectorizer`)

- **Serialization**: `pickle`

---

## Installation & Setup

Follow these steps to get the Netflix Content Discovery App up and running on your local machine.

### Prerequisites

- Python 3.8 or higher

- `pip` (Python package installer)

### 1. Clone the Repository

```bash
git clone <your_repository_url>
cd NetflixMLApp
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

#### On Windows

```bash
.\venv\Scripts\activate
```

#### On macOS/Linux

```bash
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

_(If you haven't created a `requirements.txt` yet, you can do so by running `pip freeze > requirements.txt` after installing all necessary libraries manually.)_

### 4. Run the codes

- Navigate to the `notebooks` directory and run the Jupyter notebooks `dataPreprocessing.ipynb` then run `EDA.ipynb`

- Run `eda_plots.py` and then run `finalized_dataset.py`

### 5. Build Machine Learning Models

Before running the Streamlit app, you need to train and save the genre prediction and content recommendation models. Navigate to the `scripts` directory and run the respective Python files.

```bash
python scripts/genre_prediction_model.py
python scripts/content_recommender.py
```

_(These scripts will create a `models/` directory in your project root, containing `genre_model.pkl`, `vectorizer.pkl`, `mlb.pkl`, `recommender_vectorizer.pkl`, `similarity_matrix.pkl`, and `recommender_df.csv`.)_

### 6. Run the Streamlit App

Navigate back to the root directory of your project and launch the Streamlit application.

```bash
streamlit run app.py
```

Your web browser will automatically open the Streamlit app, usually at `http://localhost:8501`.

---

## Project Structure

```markdown
NetflixMLApp/
├── dataset/
│ └── final_cleaned_dataset.csv # The core dataset
├── models/
│ ├── genre_model.pkl # Trained genre prediction model
│ ├── vectorizer.pkl # TF-IDF vectorizer for genre prediction
│ ├── mlb.pkl # MultiLabelBinarizer for genre prediction
│ ├── recommender_vectorizer.pkl # TF-IDF vectorizer for content recommendation
│ ├── similarity_matrix.pkl # Cosine similarity matrix for recommendation
│ └── recommender_df.csv # DataFrame for recommendation (with 'tags')
├── scripts/
│ ├── eda_plots.py # Functions for EDA visualizations
│ ├── genre_prediction_model.py # Script to train & save genre prediction model
│ └── content_recommender.py # Script to build & save recommendation model
├── pages/
│ ├── 1_EDA_Dashboard.py # Streamlit page for EDA
│ ├── 2_Content_Recommendation.py # Streamlit page for content recommendation
│ └── 3_Genre_Prediction.py # Streamlit page for genre prediction
├── app.py # Main Streamlit application entry point (Home page)
├── requirements.txt # List of Python dependencies
└── README.md # This file
```

---

## Credits

- **Dataset**: [Netflix Titles](https://www.kaggle.com/datasets/shivamb/netflix-shows?resource=download) (sourced from Kaggle)

- **Libraries**: Special thanks to the developers of Streamlit, Scikit-learn, Pandas, and NumPy for their incredible open-source tools.

---
