import torch
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
MODEL_PATH = Path("artifacts/transformer")
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Fine-tuned model not found. "
        "Run task4_transformers.py first."
    )
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Loading the fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH)
)
model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_PATH)
)
model.to(device)
model.eval()
print("Model loaded successfully!")
app = FastAPI(
    title="Customer Feedback Classifier",
    description="Classify customer feedback using DistilBERT.",
    version="0.1.0"
)
class PredictionRequest(BaseModel):
    text: str = Field(
        min_length=1,
        description="Customer feedback text"
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        if not value.strip():
            raise ValueError("Text cannot be empty.")

        return value

@app.get("/")
def home():
    return {
        "message": "Customer Feedback Classifier is running",
        "docs": "/docs"
    }
@app.post("/predict")
def predict(request: PredictionRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }
    with torch.no_grad():
        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]
        predicted_id = int(
            torch.argmax(probabilities).item()
        )
        confidence = float(
            probabilities[predicted_id].item()
        )
    label = model.config.id2label.get(
        predicted_id,
        str(predicted_id)
    )
    return {
        "label": label,
        "confidence": round(confidence, 4)
    }
