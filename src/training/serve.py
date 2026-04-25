from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from feast import FeatureStore

from src.schemas import (
    FEATURE_COLS,
    FeatureLookupResponse,
    PredictionResponse,
    WineClass,
    WineFeatures,
)

# point to your local MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

app = FastAPI(title="Wine Classifier API", version="1.0.0")

# load the latest staging model from MLflow registry
model = None

def load_model():
    global model
    try:
        model = mlflow.sklearn.load_model("models:/wine-classifier@staging")
        print("Model loaded from MLflow registry: Staging")
    except Exception as e:
        print(f"No staging model found, loading latest version: {e}")
        model = mlflow.sklearn.load_model("models:/wine-classifier/1")

def get_online_features(wine_id: int) -> pd.DataFrame:
    # connect to Feast feature store (make sure it's running locally)
    store = FeatureStore(repo_path="feature_store")
    feature_vector = store.get_online_features(
        features=[f"wine_features:{col}" for col in FEATURE_COLS],
        entity_rows=[{"wine_id": wine_id}]
    ).to_df()       

    print(feature_vector.columns.tolist())

    return feature_vector[FEATURE_COLS]

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="Wine Classifier API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/predict-by-id/{wine_id}", response_model=FeatureLookupResponse)
def predict_by_id(wine_id: int):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if wine_id < 0 or wine_id > 177:
        raise HTTPException(status_code=400, detail="wine_id must be between 0 and 177")

    features_df = get_online_features(wine_id)

    if features_df.empty:
        raise HTTPException(status_code=404, detail=f"No features found for wine_id {wine_id}")

    prediction = int(model.predict(features_df)[0])

    return FeatureLookupResponse(
        wine_id=wine_id,
        prediction=prediction,
        wine_class=WineClass.from_prediction(prediction),
        feature_source="feast_online_store"
    )

@app.post("/predict", response_model=PredictionResponse)
def predict(features: WineFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # convert input to dataframe
    input_df = pd.DataFrame([features.model_dump()])

    prediction = int(model.predict(input_df)[0])

    print("AFTER rename:", input_df.columns.tolist())

    return PredictionResponse(
        prediction=prediction,
        wine_class=WineClass.from_prediction(prediction),
        model_version="staging"
    )

@app.get("/reload-model")
def reload_model():
    load_model()
    return {"status": "model reloaded"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)