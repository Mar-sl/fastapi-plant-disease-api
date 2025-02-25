from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io
import torchvision.transforms as transforms
from model import load_model  # Import model from model.py


app = FastAPI()

model = load_model("model_V2.pt")

disease_info = {
    "Apple Scab": "Fungal disease. Use fungicides and remove infected leaves.",
    "Apple Black Rot": "Caused by fungus. Prune infected branches and apply fungicide.",
    "Cedar Apple Rust": "Use resistant varieties and fungicides.",
    "Healthy Apple": "No action needed. Keep monitoring.",
    "Healthy Blueberry": "No action needed.",
    "Cherry Powdery Mildew": "Apply sulfur-based fungicides.",
    "Healthy Cherry": "No action needed.",
    "Corn Gray Leaf Spot": "Rotate crops, use resistant varieties, and fungicides.",
    "Corn Common Rust": "Fungicides can help; plant resistant varieties.",
    "Corn Northern Leaf Blight": "Use resistant hybrids and foliar fungicides.",
    "Healthy Corn": "No action needed.",
    "Grape Black Rot": "Remove infected fruit, prune vines, and apply fungicides.",
    "Grape Esca": "No cure, but pruning and proper care reduce spread.",
    "Grape Leaf Blight": "Use fungicides and remove infected leaves.",
    "Healthy Grape": "No action needed.",
    "Orange Citrus Greening": "No cure. Remove infected trees to prevent spread.",
    "Peach Bacterial Spot": "Use copper-based sprays and resistant varieties.",
    "Healthy Peach": "No action needed.",
    "Pepper Bacterial Spot": "Rotate crops, use copper-based sprays.",
    "Healthy Pepper": "No action needed.",
    "Potato Early Blight": "Use fungicides and proper crop rotation.",
    "Potato Late Blight": "Destroy infected plants, apply fungicides.",
    "Healthy Potato": "No action needed.",
    "Raspberry Healthy": "No action needed.",
    "Soybean Healthy": "No action needed.",
    "Squash Powdery Mildew": "Use sulfur-based fungicides and proper spacing.",
    "Strawberry Leaf Scorch": "Remove infected leaves, improve air circulation.",
    "Healthy Strawberry": "No action needed.",
    "Tomato Bacterial Spot": "Use copper-based sprays, remove infected plants.",
    "Tomato Early Blight": "Use fungicides and proper crop rotation.",
    "Tomato Late Blight": "Destroy infected plants, apply fungicides.",
    "Tomato Leaf Mold": "Improve ventilation, use resistant varieties.",
    "Tomato Septoria Leaf Spot": "Apply fungicides and remove infected leaves.",
    "Tomato Spider Mites": "Use neem oil or insecticidal soap.",
    "Tomato Target Spot": "Apply fungicides and remove infected leaves.",
    "Tomato Yellow Leaf Curl": "Spread by whiteflies; use insect control methods.",
    "Tomato Mosaic Virus": "No cure; remove infected plants.",
    "Healthy Tomato": "No action needed."
}
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet18 expects 224x224 images
    transforms.ToTensor(),
])

def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output,dim=1).item()

    predicted_disease = list(disease_info.keys())[prediction]
    treatment_guide = disease_info[predicted_disease]

    return predicted_disease , treatment_guide


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    disease, treatment = predict_image(image_bytes)
    return {"Disease: ":disease, "Treatment: ": treatment }









