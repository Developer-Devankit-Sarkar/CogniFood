import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load model and preprocessors
model = joblib.load("model.pkl")
le_food = joblib.load("le_food.pkl")
le_category = joblib.load("le_category.pkl")
le_storage = joblib.load("le_storage.pkl")
scaler = joblib.load("scaler.pkl")

# FastAPI app
app = FastAPI()

# ✅ Allow requests from your frontend domain
origins = [
    "https://cognifood.ecostructinnovators.unaux.com",  # your frontend
    "http://localhost:8000",  # local testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,     # only allow your frontend + localhost
    allow_credentials=True,
    allow_methods=["*"],       # allow all methods (GET, POST, etc.)
    allow_headers=["*"],       # allow all headers
)

# Input schema
class FoodItem(BaseModel):
    food: str
    category: str
    storage: str

@app.get("/")
def home():
    return {"message": "Shelf Life Prediction API is running!"}

@app.post("/predict")
def predict(data: FoodItem):
    # Convert input into DataFrame
    new_food = pd.DataFrame({
        "Food Item": [data.food],
        "Category": [data.category],
        "Storage Method": [data.storage]
    })

    # Encode categorical values
    new_food["Food Item Enc"] = le_food.transform(new_food["Food Item"])
    new_food["Category Enc"] = le_category.transform(new_food["Category"])
    new_food["Storage Method Enc"] = le_storage.transform(new_food["Storage Method"])

    # Scale features
    X_new = new_food[["Food Item Enc", "Category Enc", "Storage Method Enc"]]
    X_new_scaled = scaler.transform(X_new)

    # Predict
    predicted_shelf_life = model.predict(X_new_scaled)[0]

    return {
        "food": data.food,
        "category": data.category,
        "storage": data.storage,
        "predicted_shelf_life_days": float(predicted_shelf_life)
    }
