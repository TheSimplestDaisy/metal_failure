import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import json
import os

# ========== Load Class Names ==========
class_names = ["aluminum", "steel"]  # atau load dari JSON

# ========== Load Model ==========
@st.cache_resource
def load_model():
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load("metal_fracture_classifier.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ========== Image Transform ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ========== Streamlit UI ==========
st.title("🔩 Metal Fracture Classifier")
uploaded_file = st.file_uploader("Upload an image of fractured metal", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    st.success(f"✅ Predicted Metal Type: **{predicted_class}**")
