from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import torchvision.models as models
from torch import nn

# -------------------------------
# App setup
# -------------------------------
app = Flask(__name__)
CORS(app)

# Reduce CPU usage (important for Render free tier)
torch.set_num_threads(1)

device = torch.device("cpu")

# -------------------------------
# MODEL LOADING (AUTO DETECT)
# -------------------------------
MODEL_PATH = "malaria_model.pth"

def load_model():
    try:
        # Try MobileNet first (lightweight)
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 2)

        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ Loaded MobileNet model")

    except Exception as e:
        print("⚠️ MobileNet load failed, trying ResNet...", str(e))

        # Fallback to ResNet
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)

        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("✅ Loaded ResNet model")
        except Exception as e:
            print("❌ Model loading failed, using random weights:", str(e))

    return model


# Load model safely
if os.path.exists(MODEL_PATH):
    model = load_model()
else:
    print("⚠️ Model file not found, using empty MobileNet")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)

model = model.to(device)
model.eval()

# -------------------------------
# Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

classes = ['Parasitized', 'Uninfected']

# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def home():
    return "Malaria Detection API is running"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)

            # Prediction
            _, predicted = torch.max(output, 1)

            # Confidence
            probabilities = torch.softmax(output, dim=1)
            confidence = probabilities[0][predicted.item()].item()

        return jsonify({
            "prediction": classes[predicted.item()],
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# Local run (ignored in Render)
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)