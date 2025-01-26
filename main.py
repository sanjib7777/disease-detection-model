from fastapi import FastAPI, UploadFile, File
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import io

app = FastAPI()

# Load model and processor from local directory
MODEL_DIR = "model/plant_disease_model"
processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
model.eval()

@app.get("/")
def home():
    return {"message": "Plant Disease Detection API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")

        # Preprocess image
        inputs = processor(images=image, return_tensors="pt")

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract predictions
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]
        confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_idx].item()

        return {"class": predicted_class, "confidence": round(confidence, 2)}
    
    except Exception as e:
        return {"error": str(e)}
