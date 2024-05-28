import os  # Manejo de operaciones del sistema de archivos
import torch  # Biblioteca principal para computación de tensores y redes neuronales
import gzip  # Compresión y descompresión de archivos
import cv2  # OpenCV para operaciones con imágenes
import numpy as np  # Biblioteca para operaciones con matrices
from PIL import Image  # Manejo de operaciones de imagen

# Configuración
preprocessed_dir = (
    "./preprocessed_images"  # Directorio donde se encuentran las imágenes preprocesadas
)
emotion_map = {
    0: "angry",
    1: "disgusted",
    2: "fearful",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprised",
}

# Diccionario inverso para mapear nombres de emociones a índices
emotion_map_reverse = {v: k for k, v in emotion_map.items()}


# Función para cargar tensores comprimidos
def load_tensor_compressed(filepath):
    with gzip.GzipFile(filepath, "rb") as f:  # Abrir archivo comprimido para lectura
        return torch.load(f)  # Cargar tensor desde el archivo


# Función para convertir tensores a imágenes OpenCV
def tensor_to_cv2_image(tensor):
    tensor = tensor.cpu().numpy()  # Convertir tensor a numpy array
    tensor = np.transpose(tensor, (1, 2, 0))  # Cambiar el orden de las dimensiones
    tensor = (tensor * 255).astype(np.uint8)  # Convertir a uint8
    return tensor  # Devolver la imagen en formato OpenCV


# Función para guardar una imagen temporalmente y abrirla
def save_and_open_image(img, filename):
    cv2.imwrite(filename, img)  # Guardar la imagen temporalmente
    Image.open(
        filename
    ).show()  # Abrir la imagen con el visor de imágenes predeterminado


# Función para visualizar una imagen específica
def visualizar_imagen():
    print("Seleccione el sentimiento:")
    for (
        idx,
        emotion,
    ) in emotion_map.items():  # Mostrar las opciones de emociones disponibles
        print(f"{idx}: {emotion}")

    try:
        emotion_idx = int(
            input("Ingrese el número del sentimiento: ")
        )  # Solicitar al usuario que seleccione una emoción
        if emotion_idx not in emotion_map:
            print(
                "Sentimiento inválido."
            )  # Mensaje de error si el sentimiento no es válido
            return

        emotion = emotion_map[emotion_idx]
        emotion_dir = os.path.join(preprocessed_dir, emotion)
        if not os.path.exists(
            emotion_dir
        ):  # Verificar si existen imágenes para la emoción seleccionada
            print(f"No hay imágenes preprocesadas para el sentimiento: {emotion}")
            return

        num_images = len(os.listdir(emotion_dir))
        if num_images == 0:  # Verificar si hay imágenes en la carpeta de la emoción
            print(f"No hay imágenes preprocesadas para el sentimiento: {emotion}")
            return

        print(f"Seleccione un número de imagen entre 1 y {num_images}:")
        image_idx = int(
            input("Ingrese el número de la imagen: ")
        )  # Solicitar al usuario que seleccione una imagen específica
        if image_idx < 1 or image_idx > num_images:
            print(
                "Número de imagen inválido."
            )  # Mensaje de error si el número de imagen no es válido
            return

        img_name = os.listdir(emotion_dir)[image_idx - 1]
        img_path = os.path.join(emotion_dir, img_name)
        img_tensor = load_tensor_compressed(
            img_path
        )  # Cargar el tensor comprimido de la imagen seleccionada

        # Convertir el tensor a imagen OpenCV
        img = tensor_to_cv2_image(img_tensor)

        temp_filename = "temp_image.png"
        save_and_open_image(
            img, temp_filename
        )  # Guardar y abrir la imagen temporalmente

    except ValueError:
        print(
            "Entrada inválida. Por favor, ingrese un número."
        )  # Mensaje de error si la entrada no es un número


if __name__ == "__main__":
    visualizar_imagen()  # Ejecutar la función principal para visualizar la imagen
