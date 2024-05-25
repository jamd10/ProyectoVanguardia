from flask import Flask, request, jsonify, render_template
import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import os

app = Flask(__name__)

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "AI/best_model.pth"

# Cargar el modelo
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 7)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Transforms para preprocesar las imágenes
data_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

emotion_map = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]


# Función para detectar y recortar el rostro en la imagen
def detect_face(image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return None

    # Solo tomamos la primera cara detectada
    x, y, w, h = faces[0]
    face = image.crop((x, y, x + w, y + h))
    return face


def preprocess_image(image):
    face = detect_face(image)
    if face is None:
        return None
    face = data_transforms(face)
    face = face.unsqueeze(0)
    return face.to(device)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(file).convert("RGB")
        image = preprocess_image(image)
        if image is None:
            return jsonify({"error": "No face detected"}), 400
        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            sentiment = emotion_map[preds.item()]
        return jsonify({"sentiment": sentiment})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
