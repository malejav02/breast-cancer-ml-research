from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import time
from typing import Union, List, Dict
from src.utils.paths import models_dir

app = FastAPI()

# load model
model_dict = joblib.load(models_dir("wisconsin_best_model.pkl"))
model = model_dict["model"]
selected_features = model_dict["selected_features"]


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict(data: Union[Dict, List[Dict]]):
    """
    Supports:
    - Single sample: dict
    - Batch: list of dicts
    """

    try:
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)

        df = df[selected_features]

        if df.isnull().values.any():
            nan_cols = df.columns[df.isnull().any()].tolist()
            raise HTTPException(
                status_code=400,
                detail=f"Input data contains NaN values in columns: {nan_cols}",
            )

        start = time.time()
        predictions = model.predict(df)
        total_latency = time.time() - start
        latency_per_sample = total_latency / len(df)

        labels = ["malignant" if pred == 0 else "benign" for pred in predictions]

        return {
            "n_samples": len(df),
            "predictions": predictions.tolist(),
            "labels": labels,
            "latency_total_seconds": total_latency,
            "latency_per_sample_seconds": latency_per_sample,
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        return {"error": str(e)}
