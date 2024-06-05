import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import joblib
import os

def load_models():
    model_dir = os.path.dirname(__file__)
    kmeans_model = joblib.load(os.path.join(model_dir, 'Models/kmeans.pkl'))
    dbscan_model = joblib.load(os.path.join(model_dir, 'Models/dbscan.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'Models/scaler.pkl'))
    return kmeans_model, dbscan_model, scaler

try:
    kmeans_model, dbscan_model, scaler = load_models()
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

# Streamlit app title
st.set_page_config(page_title='Goodreads Book Analysis', page_icon=':books:', layout='wide', initial_sidebar_state='auto')
st.title('Goodreads Book Analysis')

df = pd.read_csv('/Users/raneemaj/Documents/GitHub/UC8/goodreads.csv')
df.drop_duplicates(inplace=True)

# Visualizing rating distribution
st.write('Visualizing rating distribution...')
fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(df['Rating'], bins=20, kde=False, ax=ax, color='#4CAF50')
ax.set_title('Rating Distribution', fontsize=16, fontweight='bold')
ax.set_xlabel('Rating', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)
st.pyplot(fig)

# Load ratings count dataset
df_counts = pd.read_csv('Ratings_count.csv')
df_counts.rename(columns={'Book_titles': 'Book_Title'}, inplace=True)
df_counts['Book_Title'] = df_counts['Book_Title'].str.split('(').str[0].str.strip()

# Preprocess genres
df = df[df['Genres'].str.len() != 2]
df['Genres'] = df['Genres'].str.replace('[', '').str.replace(']', '').str.replace("'", '').str.split(', ')
df3 = df.explode('Genres')

# Select top genres
selected_genres = df3['Genres'].value_counts().index[:20]
df4 = df3[df3['Genres'].isin(selected_genres)]
df5 = df4.groupby('Book_Title').agg({
    'Genres': list,
    'Rating': 'mean'
}).reset_index()

# Create genre columns
def has_genre(genres, genre):
    return 1 if genre in genres else 0

for genre in selected_genres:
    df5[genre] = df5['Genres'].apply(lambda x: has_genre(x, genre))

# Merge data
df6 = df5.merge(df_counts, on='Book_Title', how='inner')

# Visualizing genres distribution
st.write('Visualizing genres distribution...')
genres_count = df4['Genres'].value_counts()
fig, ax = plt.subplots(figsize=(24, 6))
sns.barplot(x=genres_count.index, y=genres_count.values, ax=ax, palette='Set2')
ax.set_title('Genres Distribution', fontsize=16, fontweight='bold')
ax.tick_params(axis='x', rotation=90)
st.pyplot(fig)

# Top 10 Books by Rating Count
st.write('Top 10 Books by Rating Count')
most_rated = df6.sort_values('Ratings_count', ascending=False).head(10).set_index('Book_Title')
fig, ax = plt.subplots(figsize=(15, 10))
sns.barplot(x=most_rated['Ratings_count'], y=most_rated.index, palette='Oranges', ax=ax)
ax.set_xlabel('Rating Count', fontsize=14)
ax.set_ylabel('Book Title', fontsize=14)
ax.set_title('Top 10 Books by Rating Count', fontsize=16, fontweight='bold')
st.pyplot(fig)

# Top 10 Authors with Highly Rated Books
st.write('Top 10 Authors with Highly Rated Books')
high_rated_author = df[df['Rating'] >= 4.3]
high_rated_author = high_rated_author.groupby('Author')['Book_Title'].count().reset_index().sort_values('Book_Title', ascending=False).head(10).set_index('Author')
fig, ax = plt.subplots(figsize=(15, 10))
sns.barplot(x=high_rated_author['Book_Title'], y=high_rated_author.index, palette='Purples', ax=ax)
ax.set_xlabel('Number of Books', fontsize=14)
ax.set_ylabel('Authors', fontsize=14)
ax.set_title('Top 10 Authors with Highly Rated Books', fontsize=16, fontweight='bold')
st.pyplot(fig)

# Sidebar for clustering options
st.sidebar.title("Clustering Options")
option = st.sidebar.selectbox(
    "Choose the section:",
    ("Home", "KMeans Clustering", "DBSCAN Clustering")
)

category_mapping = {genre: i for i, genre in enumerate(selected_genres)}
reverse_category_mapping = {i: genre for genre, i in category_mapping.items()}

if option == "Home":
    st.title("Book Clustering Prediction")
    st.header("Training Phase Visualizations")

elif option == "KMeans Clustering":
    st.title("KMeans Clustering")
    genre = st.sidebar.selectbox("Select Genre", list(category_mapping.keys()))
    genre_encoded = category_mapping[genre]

    if st.sidebar.button("Predict KMeans"):
        response = requests.post("https://uc8-r6r0.onrender.com/predict/kmeans", json={
            "Genre_encoded": genre_encoded
        })
        result = response.json()
        if "cluster" in result:
            st.write(f"KMeans Cluster: {result['cluster']}")
        else:
            st.write("KMeans Cluster: Not assigned to any cluster")

elif option == "DBSCAN Clustering":
    st.title("DBSCAN Clustering")
    genre = st.sidebar.selectbox("Select Genre", list(category_mapping.keys()))
    genre_encoded = category_mapping[genre]

    if st.sidebar.button("Predict DBSCAN"):
        response = requests.post("https://uc8-r6r0.onrender.com/predict/dbscan", json={
            "Genre_encoded": genre_encoded
        })
        result = response.json()
        if "cluster" in result:
            st.write(f"DBSCAN Cluster: {result['cluster']}")
        else:
            st.write("DBSCAN Cluster: Not assigned to any cluster")

            response = requests.post("https://uc8-r6r0.onrender.com/predict/kmeans", json={"Genre_encoded": genre_encoded})
        print(response.json())  # Check what the server is actually returning