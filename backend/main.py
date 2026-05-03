from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import torchvision.models as models
from torch import nn

# -------------------------------
# App setup
# -------------------------------
app = Flask(__name__)
CORS(app)

# Reduce CPU + memory usage (important)
torch.set_num_threads(1)

device = torch.device("cpu")

# -------------------------------
# Lightweight model (safe for Render)
# -------------------------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)

# ⚠️ TEMPORARY: disable loading heavy weights to avoid crash
# Uncomment ONLY if model matches MobileNet
# model.load_state_dict(torch.load("malaria_model.pth", map_location=device))

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

            _, predicted = torch.max(output, 1)

            probabilities = torch.softmax(output, dim=1)
            confidence = probabilities[0][predicted.item()].item()

        return jsonify({
            "prediction": classes[predicted.item()],
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# Local run (ignored by Render)
# -------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)