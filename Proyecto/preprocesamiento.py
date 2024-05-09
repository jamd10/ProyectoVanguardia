from PIL import Image, ImageEnhance
import os

def preprocess_image(image_path, save_dir):
    # Abre la imagen
    img = Image.open(image_path)

    # Normaliza la imagen (escala de grises)
    img = img.convert('L')

    # Redimensiona la imagen
    img = img.resize((300, 300))

    # Ajusta el brillo
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.2)  # Aumenta el brillo en un 20%

    # Crea el directorio si no existe
    os.makedirs(save_dir, exist_ok=True)

    # Guarda la imagen preprocesada
    base_name = os.path.basename(image_path)
    save_path = os.path.join(save_dir, "preprocessed_" + base_name)
    img.save(save_path)

    return img

# Diccionario con los nombres de las carpetas y la cantidad de imágenes en cada una
folders = {
    "angry": 3995,
    "disgusted": 436,
    "fearful": 4097,
    "happy": 7215,
    "neutral": 4965,
    "sad": 4830,
    "surprised": 3171
}

# Ruta al directorio de las imágenes
image_dir = r"C:\Users\jamdj\OneDrive\Escritorio\Proyecto Vanguardia\archive\train"

# Ruta al directorio donde se guardarán las imágenes preprocesadas
save_dir = r"C:\Users\jamdj\OneDrive\Escritorio\Proyecto Vanguardia\preprocessed_images"

# Procesa todas las imágenes en cada carpeta
for folder, num_images in folders.items():
    for i in range(num_images):
        image_path = os.path.join(image_dir, folder, f"im{i}.png")
        folder_save_dir = os.path.join(save_dir, folder)
        preprocess_image(image_path, folder_save_dir)

print("¡Imágenes preprocesadas correctamente!")
