import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Function to load models and scaler
def load_models():
    model_dir = os.path.dirname(__file__)
    kmeans_model = joblib.load(os.path.join(model_dir, 'models/kmeans_model.pkl'))
    dbscan_model = joblib.load(os.path.join(model_dir, 'models/dbscan_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'models/scaler.pkl'))
    return kmeans_model, dbscan_model, scaler

# Load models and scaler
try:
    kmeans_model, dbscan_model, scaler = load_models()
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

# Mapping of Category_encoded to Category names
category_mapping = {
    8: "Café",
    4: "Bakery",
    5: "Breakfast",
    9: "Donuts",
    7: "Burgers",
    25: "Restaurant",
    27: "Sandwich Spot",
    23: "Pizza",
    22: "Middle Eastern",
    19: "Japanese",
    29: "Shawarma Restaurant",
    31: "Steakhouse",
    12: "Fast Food",
    18: "Italian",
    17: "Indian",
    26: "Salad",
    14: "Food Truck",
    15: "French",
    1: "Armenian",
    13: "Food Court",
    21: "Mediterranean",
    30: "Snacks",
    20: "Lebanese",
    3: "BBQ",
    11: "Falafel",
    16: "Fried Chicken",
    24: "Poke Restaurant",
    33: "Swiss",
    32: "Sushi",
    34: "Vegan and Vegetarian Restaurant",
    6: "Buffet",
    28: "Seafood",
    0: "American",
    2: "Asian",
    10: "Eastern European"
}

# Reverse mapping for dropdown display
reverse_category_mapping = {v: k for k, v in category_mapping.items()}

# Sidebar options
option = st.sidebar.selectbox(
    "Choose the section:",
    ("Home", "KMeans Clustering", "DBSCAN Clustering")
)

if option == "Home":
    st.title("Restaurant Clustering Prediction")
    st.write("""
        ## Welcome to the Restaurant Clustering Prediction App
        This application allows you to predict restaurant clusters using KMeans and DBSCAN clustering methods. 
        Explore the visualizations from the training phase and then switch to the models to make predictions.
    """)

    st.header("Training Phase Visualizations")

    # Elbow Method for KMeans
    st.subheader("Elbow Method for KMeans")
    wcss_img_path = 'model-deployment/imgs/wcss.png'
    st.image(wcss_img_path, caption='Elbow Method for KMeans')

    # K-Distance Graph for DBSCAN
    st.subheader("K-Distance Graph for DBSCAN")
    k_dist_sorted_img_path = 'model-deployment/imgs/k-distance.png'
    st.image(k_dist_sorted_img_path, caption='K-Distance Graph for DBSCAN')

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    try:
        df = pd.read_csv("Data/Cleand_data.csv")  
        df.drop('Unnamed: 0', axis=1, inplace=True)
        correlation = df.corr(numeric_only=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(round(correlation, 2), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        st.pyplot(plt)
    except FileNotFoundError:
        st.error("Cleand_data.csv file not found. Please make sure the file is in the 'Data' directory.")

elif option == "KMeans Clustering":
    st.title("KMeans Clustering")
    score = st.number_input("Enter Score", min_value=0.0, max_value=10.0, value=9.0)
    price_range = st.selectbox("Select Price Range", ["$", "$$", "$$$"])
    price_range_encoded = {"$": 0, "$$": 1, "$$$": 2}[price_range]
    category = st.selectbox("Select Category", list(reverse_category_mapping.keys()))
    category_encoded = reverse_category_mapping[category]

    if st.button("Predict KMeans"):
        response = requests.post("https://project-5-itbj.onrender.com/predict/kmeans", json={
            "Score": score, 
            "Price_Range_encoded": price_range_encoded,
            "Category_encoded": category_encoded
        })
        result = response.json()
        if "cluster" in result:
            st.write(f"KMeans Cluster: {result['cluster']}")
        else:
            st.write("KMeans Cluster: Not assigned to any cluster")

elif option == "DBSCAN Clustering":
    st.title("DBSCAN Clustering")
    score = st.number_input("Enter Score", min_value=0.0, max_value=10.0, value=9.0)
    price_range = st.selectbox("Select Price Range", ["$", "$$", "$$$"])
    price_range_encoded = {"$": 0, "$$": 1, "$$$": 2}[price_range]
    category = st.selectbox("Select Category", list(reverse_category_mapping.keys()))
    category_encoded = reverse_category_mapping[category]

    if st.button("Predict DBSCAN"):
        response = requests.post("https://project-5-itbj.onrender.com/predict/dbscan", json={
            "Score": score, 
            "Price_Range_encoded": price_range_encoded,
            "Category_encoded": category_encoded
        })
        result = response.json()
        if "cluster" in result:
            st.write(f"DBSCAN Cluster: {result['cluster']}")
        else:
            st.write("DBSCAN Cluster: Not assigned to any cluster")
