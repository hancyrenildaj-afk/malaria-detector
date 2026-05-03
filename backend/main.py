from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import torchvision
from torch import nn

app = Flask(__name__)
CORS(app)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture
model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# Load trained weights
model.load_state_dict(torch.load("malaria_model.pth", map_location=device))

model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

classes = ['Parasitized', 'Uninfected']


@app.route('/')
def home():
    return "Malaria Detection API is running"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        return jsonify({
            "prediction": classes[predicted.item()]
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# 🔥 IMPORTANT: outside function + correct indentation
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)