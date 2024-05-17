import os
import torch
from torchvision import transforms
from PIL import Image

# Define los directorios de las imágenes
root_dir = "archive/train"
folders = {
    "angry": 3995,
    "disgusted": 436,
    "fearful": 4097,
    "happy": 7215,
    "neutral": 4965,
    "sad": 4830,
    "surprised": 3171
}

# Define transformación inversa para convertir el tensor a imagen
transform_inverse = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),  # Inversa de la normalización
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),  # Inversa de la normalización
    transforms.ToPILImage()  # Convierte el tensor a imagen PIL
])

print("Elija un número para el sentimiento:")
for i, emotion in enumerate(folders.keys(), start=1):
    print(f"{i}. {emotion}")

sentimiento = int(input())
emotion = list(folders.keys())[sentimiento - 1]

print(f"Elija un número entre 0 y {folders[emotion]}")
numero = int(input())

image_path = f"preprocessed_images/{emotion}/im{numero}.pt"
image_tensor = torch.load(image_path)

# Aplica la transformación inversa para obtener la imagen
image = transform_inverse(image_tensor)

image.show()
