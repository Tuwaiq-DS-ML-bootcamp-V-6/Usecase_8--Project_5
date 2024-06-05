from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI()

# Load your model
model_dir = os.path.dirname(__file__)
kmeans_model = joblib.load(os.path.join(model_dir, 'models/kmeans_model.pkl'))
dbscan_model = joblib.load(os.path.join(model_dir, 'models/dbscan_model.pkl'))
scaler = joblib.load(os.path.join(model_dir, 'models/scaler.pkl'))

class PredictionRequest(BaseModel):
    Score: float
    Price_Range_encoded: int
    Category_encoded: int

@app.post("/predict/kmeans")
async def predict_kmeans(data: PredictionRequest):
    df = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    scaled_data = scaler.transform(df)
    prediction = kmeans_model.predict(scaled_data)
    return {"cluster": int(prediction[0])}

@app.post("/predict/dbscan")
async def predict_dbscan(data: PredictionRequest):
    df = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    scaled_data = scaler.transform(df)
    prediction = dbscan_model.fit_predict(scaled_data)
    cluster_label = int(prediction[0]) if prediction.size > 0 else -1  # Handle case where no cluster is assigned
    return {"cluster": cluster_label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
