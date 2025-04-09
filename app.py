# app.py
# import the required liberaries

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from torchvision import transforms
from timm import create_model
import matplotlib.pyplot as plt

# Constants
CLASSES = ["healthy", "multiple_diseases", "rust", "scab"]
NUM_CLASSES = len(CLASSES)
MODEL_PATH = "client1_final_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
@st.cache_resource
def load_model():
    model = create_model("tf_efficientnet_b0.ns_jft_in1k", pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def load_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def predict(model, image):
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1).squeeze().cpu().numpy()
    return probs

def display_result(image, probs):
    pred_idx = int(np.argmax(probs))
    pred_label = CLASSES[pred_idx]
    confidence = probs[pred_idx]

    st.image(image, caption=f"Prediction: {pred_label} ({confidence:.2%})", use_column_width=True)

    # Bar chart
    fig, ax = plt.subplots()
    ax.barh(CLASSES, probs, color='lightgreen')
    ax.set_xlim(0, 1)
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i, f"{v:.2%}", va='center')
    ax.set_title("Class Probabilities")
    st.pyplot(fig)

# Streamlit UI
st.set_page_config(page_title="Apple Leaf Classifier 🍏", layout="wide")
st.title("🍃 Apple Leaf Disease Classifier")
st.write("Upload a leaf image (or a folder of images) to predict health status.")

model = load_model()

uploaded_files = st.file_uploader("Choose image(s)", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files:
    all_results = []

    for file in uploaded_files:
        image = Image.open(file).convert("RGB")
        tensor = load_image(image)
        probs = predict(model, tensor)
        display_result(image, probs)

        result = {
            "filename": file.name,
            "predicted_label": CLASSES[np.argmax(probs)],
            "confidence": probs[np.argmax(probs)],
            **{f"prob_{cls}": prob for cls, prob in zip(CLASSES, probs)}
        }
        all_results.append(result)

    df = pd.DataFrame(all_results)
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False), "results.csv", "text/csv")
