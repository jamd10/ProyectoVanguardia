import os  # Manejo de operaciones del sistema de archivos
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
)  # Importar Flask y módulos necesarios para la API
import torch  # Biblioteca principal para computación de tensores y redes neuronales
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
)  # Modelos y procesadores de imágenes preentrenados de Hugging Face
from PIL import Image  # Manejo de operaciones de imagen

# Crear una instancia de Flask
app = Flask(__name__)

# Configuración del dispositivo
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Usar GPU si está disponible, de lo contrario usar CPU

# Nombre del modelo y ruta del modelo entrenado
model_name = "google/vit-base-patch16-224"  # Nombre del modelo preentrenado de Vision Transformer (ViT)
model_path = "AI/best_model.pth"  # Ruta al modelo entrenado guardado

# Mapeo de emociones de números a nombres
emotion_map = {
    0: "angry",
    1: "disgusted",
    2: "fearful",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprised",
}

# Traducción de emociones al español
emotion_translation = {
    "angry": "Enojo",
    "disgusted": "Disgusto",
    "fearful": "Miedo",
    "happy": "Felicidad",
    "neutral": "Neutral",
    "sad": "Tristeza",
    "surprised": "Sorpresa",
}

# Cargar el modelo y el feature extractor de Hugging Face
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=7,
    ignore_mismatched_sizes=True,  # Cargar el modelo con 7 etiquetas y permitir tamaños desajustados
)
if os.path.exists(model_path):  # Si existe un modelo entrenado guardado, cargarlo
    model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)  # Mover el modelo al dispositivo (GPU o CPU)
model.eval()  # Poner el modelo en modo de evaluación

feature_extractor = ViTImageProcessor.from_pretrained(
    model_name
)  # Cargar el procesador de imágenes


# Función para preprocesar una imagen usando ViTImageProcessor
def preprocess_image_vit(image):
    image = image.convert("RGB")  # Convertir imagen a RGB
    inputs = feature_extractor(
        images=image, return_tensors="pt"
    )  # Extraer características y convertir a tensores de PyTorch
    return (
        inputs["pixel_values"].squeeze(0).to(device)
    )  # Mover tensor al dispositivo (GPU o CPU)


# Ruta de la página principal
@app.route("/")
def index():
    return render_template(
        "index.html"
    )  # Renderizar plantilla HTML de la página principal


# Ruta para predecir la emoción de una imagen
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:  # Verificar si no se envió archivo
        return jsonify({"error": "No file part"})

    file = request.files["file"]  # Obtener archivo del formulario
    if (
        file.filename == ""
    ):  # Verificar si el archivo no tiene nombre (no se seleccionó archivo)
        return jsonify({"error": "No selected file"})

    try:
        image = Image.open(file)  # Abrir imagen
        img_tensor = preprocess_image_vit(image).unsqueeze(
            0
        )  # Preprocesar la imagen y añadir dimensión de lote

        with torch.no_grad():  # Desactivar cálculo de gradientes
            outputs = model(img_tensor)  # Obtener predicciones del modelo
            _, preds = torch.max(outputs.logits, 1)  # Obtener la etiqueta predicha
            emotion = emotion_map[preds.item()]  # Mapear la etiqueta a la emoción
            emotion_es = emotion_translation[emotion]  # Traducir la emoción al español

        return jsonify({"sentiment": emotion_es})  # Devolver la emoción en formato JSON
    except Exception as e:
        return jsonify({"error": str(e)})  # Devolver error en caso de excepción


# Ejecutar la aplicación Flask
if __name__ == "__main__":
    app.run(debug=True)  # Ejecutar la aplicación en modo de depuración
