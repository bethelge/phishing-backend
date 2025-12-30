from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import joblib
import numpy as np

# create FastAPI app
app = FastAPI(title="Phishing Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all domains
    allow_credentials=True,
    allow_methods=["*"],   # allow all methods (GET, POST, etc.)
    allow_headers=["*"],   # allow all headers
)
# load trained models
lr_model = joblib.load("logistic_model.joblib")
dt_model = joblib.load("decision_tree_model.joblib")

# root endpoint
@app.get("/")
def root():
    return {"message": "Phishing Detection API is running"}

# Pydantic model for input
class FeatureInput(BaseModel):
    features: List[float]

# prediction endpoint
@app.post("/predict/{model_type}")
def predict(model_type: str, input: FeatureInput):
    try:
        # convert features to numpy array
        data = np.array(input.features).reshape(1, -1)

        if model_type == "logistic":
            prediction = lr_model.predict(data)[0]
        elif model_type == "tree":
            prediction = dt_model.predict(data)[0]
        else:
            return {"error": "Invalid model_type. Use 'logistic' or 'tree'"}

        return {"prediction": "Legitimate" if prediction == 1 else "Phishing"}
    except Exception as e:
        return {"error": str(e)}
