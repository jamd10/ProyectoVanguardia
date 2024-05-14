import os
from PIL import Image
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim

# Verifica si CUDA está disponible y establece el dispositivo a GPU si es posible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define las transformaciones de preprocesamiento
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensiona las imágenes
    transforms.RandomHorizontalFlip(),  # Volteo horizontal aleatorio
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Cambio de brillo, contraste, saturación y tonalidad
    transforms.RandomRotation(15),  # Rotación aleatoria de hasta 15 grados
    transforms.ToTensor(),  # Convierte las imágenes a tensores
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización
])

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

# Crea la carpeta de imágenes preprocesadas si no existe
if not os.path.exists("preprocessed_images"):
    os.makedirs("preprocessed_images")

# Verifica si las imágenes ya han sido preprocesadas
preprocessed = all(os.path.exists(f"preprocessed_images/{emotion}/im{i}.pt") for emotion, num_images in folders.items() for i in range(num_images))

if not preprocessed:
    # Procesa las imágenes para cada sentimiento
    for emotion, num_images in folders.items():
        print(f"Procesando imágenes para {emotion}...")
        
        # Crea una subcarpeta para cada sentimiento en la carpeta de imágenes preprocesadas
        if not os.path.exists(f"preprocessed_images/{emotion}"):
            os.makedirs(f"preprocessed_images/{emotion}")
        
        for i in range(num_images):
            # Carga la imagen como una imagen PIL
            image = Image.open(f"{root_dir}/{emotion}/im{i}.png")
            
            # Convierte la imagen a RGB si es en escala de grises
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Aplica transformaciones de aumento de datos
            image = transform(image)
            
            # Guarda la imagen preprocesada
            torch.save(image, f"preprocessed_images/{emotion}/im{i}.pt")
            
    print("¡Preprocesamiento completado!")
else:
    print("Las imágenes ya han sido preprocesadas.")