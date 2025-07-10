import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os
import gdown

# === Class Labels (update if needed) ===
class_names = ["aluminum", "steel"]

# === Download model from Google Drive if not exists ===
MODEL_FILE = "metal_fracture_classifier.pt"
GDRIVE_ID = "1fGZCY53L7uQ28Wq7J57mY5vhPW8vf1sD/view?usp=drive_link"  # 🔁 Gantikan dengan ID anda
GDRIVE_URL = f"https://drive.google.com/file/d/1fGZCY53L7uQ28Wq7J57mY5vhPW8vf1sD/view?usp=drive_link"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        st.info("📥 Downloading model file...")
        gdown.download(GDRIVE_URL, MODEL_FILE, quiet=False)
    
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device("cpu")))
    model.eval()
    return model

# === Image Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Streamlit App ===
st.title("🔩 Metal Fracture Type Classifier")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    input_tensor = transform(image).unsqueeze(0)

    with st.spinner("🔍 Classifying..."):
        model = load_model()
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            pred_class = class_names[predicted.item()]
    
    st.success(f"✅ Prediction: **{pred_class}**")
