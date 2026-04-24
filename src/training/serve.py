from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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

# input schema — matches wine dataset feature names
class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

class PredictionResponse(BaseModel):
    prediction: int
    wine_class: str
    model_version: str

WINE_CLASSES = {0: "Class 0 (Cultivar 1)", 1: "Class 1 (Cultivar 2)", 2: "Class 2 (Cultivar 3)"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="Wine Classifier API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

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
        wine_class=WINE_CLASSES[prediction],
        model_version="staging"
    )

@app.get("/reload-model")
def reload_model():
    load_model()
    return {"status": "model reloaded"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)