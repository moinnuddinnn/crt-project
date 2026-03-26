import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import joblib

# Load data
df = pd.read_csv("assignment_completion_time_5000.csv")

X = df[["difficulty_level", "focus_level"]]
y = df["completion_time"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# Train
pipeline.fit(X_train, y_train)

# SAVE MODEL ✅
joblib.dump(pipeline, "model.pkl")

print("✅ Model trained and saved as model.pkl")