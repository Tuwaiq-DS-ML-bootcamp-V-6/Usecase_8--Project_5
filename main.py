from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from typing import List


app = FastAPI()

# GET request
@app.get("/")
def read_root():
    return {"message": "Welcome to My anime api"}

# get request
@app.get("/items/")
def create_item(item: dict):
    return {"item": item}

model = joblib.load('Models\kMeans_model.joblib')
scaler = joblib.load('Models\scaler.joblib')

mlb_genres= joblib.load('Models/mlb_genres.joblib')
mlb_demographics= joblib.load('Models/mlb_demographics.joblib')

# Define a Pydantic model for input data validation

class InputFeatures(BaseModel):
    Seasonal:bool
    Genre: List[str] = ["Action"]
    Demographic: List[str] = ["Kids"]

import os

def preprocessing(input_features: InputFeatures):
    one_hot_genres = pd.DataFrame(mlb_genres.transform([input_features.Genre]), columns=mlb_genres.classes_)
    

    
    one_hot_demographics = pd.DataFrame(mlb_demographics.transform([input_features.Demographic]), columns=mlb_demographics.classes_)

    dict_f = {
        'Seasonal': input_features.Seasonal,
        'Action': one_hot_genres['Action'],
        'Adventure': one_hot_genres['Adventure'],
        'AvantGarde': one_hot_genres['AvantGarde'],
        'AwardWinning': one_hot_genres['AwardWinning'],
        'Comedy': one_hot_genres['Comedy'],
        'Drama': one_hot_genres['Drama'],
        'Fantasy': one_hot_genres['Fantasy'],
        'Horror': one_hot_genres['Horror'],
        'Mystery': one_hot_genres['Mystery'],
        'Romance': one_hot_genres['Romance'],
        'Sci-Fi': one_hot_genres['Sci-Fi'],
        'SliceofLife': one_hot_genres['SliceofLife'],
        'Sports': one_hot_genres['Sports'],
        'Supernatural': one_hot_genres['Supernatural'],
        'Suspense':  one_hot_genres['Suspense'],
        'Kids': one_hot_demographics['Kids'],
        'Seinen': one_hot_demographics['Seinen'],
        "Shounen": one_hot_demographics['Shounen']
    }

    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
    return dict_f

@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    f = open("demofile2.txt", "w")
    f.write("data")
    f.flush()
    f.close()
    y_pred = model.predict(pd.DataFrame(data))
    return {"pred": y_pred.tolist()[0]}


