# test.py
# The test phase
# Import the required libraries
import os
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from timm import create_model

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "client1_final_model.pth"
CLASSES = ["healthy", "multiple_diseases", "rust", "scab"]
NUM_CLASSES = len(CLASSES)

# Load image
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # (1, C, H, W)

# Load model
def load_model():
    model = create_model("tf_efficientnet_b0.ns_jft_in1k", pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Predict single image
def predict_image(model, image_path):
    image_tensor = load_image(image_path).to(DEVICE)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1).squeeze().cpu().numpy()
    return probs

# Display results
def show_prediction(image_path, probs):
    predicted_index = int(np.argmax(probs))
    predicted_label = CLASSES[predicted_index]
    confidence = probs[predicted_index]

    image = Image.open(image_path).convert("RGB")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.imshow(image)
    ax1.axis("off")
    ax1.set_title(f"Prediction: {predicted_label}\nConfidence: {confidence:.2%}")

    ax2.barh(CLASSES, probs, color="skyblue")
    ax2.set_xlim(0, 1)
    ax2.set_title("Class Probabilities")
    for i, v in enumerate(probs):
        ax2.text(v + 0.01, i, f"{v:.2%}", va='center')

    plt.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python test.py <image.jpg> or <folder/>")
        sys.exit(1)

    path = sys.argv[1]
    model = load_model()
    results = []

    if os.path.isfile(path):
        probs = predict_image(model, path)
        show_prediction(path, probs)
        predicted_label = CLASSES[np.argmax(probs)]
        confidence = probs[np.argmax(probs)]
        results.append({
            "filename": os.path.basename(path),
            "predicted_label": predicted_label,
            "confidence": confidence,
            **{f"prob_{cls}": prob for cls, prob in zip(CLASSES, probs)}
        })

    elif os.path.isdir(path):
        images = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        for img_name in images:
            img_path = os.path.join(path, img_name)
            probs = predict_image(model, img_path)
            show_prediction(img_path, probs)
            predicted_label = CLASSES[np.argmax(probs)]
            confidence = probs[np.argmax(probs)]
            results.append({
                "filename": img_name,
                "predicted_label": predicted_label,
                "confidence": confidence,
                **{f"prob_{cls}": prob for cls, prob in zip(CLASSES, probs)}
            })
    else:
        print("Invalid path: must be image file or folder.")
        sys.exit(1)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    print("Results saved to results.csv")
