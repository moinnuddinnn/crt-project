from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

app = FastAPI(title="Student Psychology ML API")

# Enable CORS (important for deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input validation
class InputData(BaseModel):
    difficulty_level: int = Field(..., ge=1, le=10)
    focus_level: int = Field(..., ge=1, le=10)

@app.get("/")
def home():
    return {"message": "API is running 🚀"}

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.difficulty_level, data.focus_level]])
    prediction = model.predict(X)[0]

    return {
        "predicted_completion_time": round(float(prediction), 2)
    }