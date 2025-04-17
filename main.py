from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os

app = FastAPI()

model = None
vectorizer = None

@app.on_event("startup")
def load_model():
    global model, vectorizer
    MODEL_PATH = "spam_classifier.pkl"
    VECTORIZER_PATH = "vectorizer.pkl"

    print("🔍 Attempting to load model and vectorizer...")

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")
        if not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(f"❌ Vectorizer file not found at {VECTORIZER_PATH}")

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully.")

        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        print("✅ Vectorizer loaded successfully.")

    except Exception as e:
        print(f"🔥 Error during startup: {e}")
        raise e

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

class Message(BaseModel):
    message: str

@app.post("/predict")
def predict(data: Message):
    text = data.message
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    label = "spam" if int(prediction) == 1 else "not spam"
    return {"prediction": label}

