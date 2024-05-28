import os  # Manejo de operaciones del sistema de archivos
import copy  # Para hacer copias profundas de objetos
import torch  # Biblioteca principal para computación de tensores y redes neuronales
import torch.nn as nn  # Módulo para construir y entrenar modelos de redes neuronales
import torch.optim as optim  # Módulo para algoritmos de optimización
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
)  # Modelos y procesadores de imágenes preentrenados de Hugging Face
from torch.utils.data import (
    DataLoader,
    Dataset,
)  # Para manejar datos de entrenamiento y prueba
from tqdm import tqdm  # Mostrar barras de progreso
from PIL import Image  # Manejo de operaciones de imagen
from torch.cuda.amp import autocast, GradScaler  # Entrenamiento de precisión mixta
import gzip  # Compresión y descompresión de archivos
from concurrent.futures import ThreadPoolExecutor  # Manejo de concurrencia

# Configuraciones
data_dir = "./archive"  # Directorio donde se encuentran los datos originales
preprocessed_dir = (
    "./preprocessed_images"  # Directorio para almacenar imágenes preprocesadas
)
model_dir = "./AI"  # Directorio para guardar el modelo entrenado
batch_size = 64  # Tamaño del lote para entrenamiento
num_epochs = 10  # Número de épocas para el entrenamiento
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Usar GPU si está disponible, de lo contrario usar CPU
model_name = "google/vit-base-patch16-224"  # Nombre del modelo preentrenado de Vision Transformer (ViT)

# Cargar el modelo y el feature extractor de Hugging Face
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=7,
    ignore_mismatched_sizes=True,  # Cargar el modelo con 7 etiquetas y permitir tamaños desajustados
)
model.to(device)  # Mover el modelo al dispositivo (GPU o CPU)
feature_extractor = ViTImageProcessor.from_pretrained(
    model_name
)  # Cargar el procesador de imágenes

# Diccionario para mapear etiquetas de emociones a números
emotion_map = {
    "angry": 0,
    "disgusted": 1,
    "fearful": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprised": 6,
}


# Función para preprocesar una imagen usando ViTImageProcessor
def preprocess_image_vit(image_path):
    image = Image.open(image_path).convert("RGB")  # Abrir imagen y convertir a RGB
    inputs = feature_extractor(
        images=image, return_tensors="pt"
    )  # Extraer características y convertir a tensores de PyTorch
    return (
        inputs["pixel_values"].squeeze(0).half()
    )  # Convertir a float16 para reducir el uso de memoria


# Función para guardar tensores comprimidos
def save_tensor_compressed(tensor, filepath):
    with gzip.GzipFile(filepath, "wb") as f:  # Abrir archivo comprimido para escritura
        torch.save(
            tensor, f, pickle_protocol=4
        )  # Guardar tensor con protocolo de pickle 4


# Función para cargar tensores comprimidos
def load_tensor_compressed(filepath):
    with gzip.GzipFile(filepath, "rb") as f:  # Abrir archivo comprimido para lectura
        return torch.load(f)  # Cargar tensor desde el archivo


# Función para preprocesar y guardar una sola imagen
def preprocess_and_save_image(img_path, preprocessed_img_path):
    img_tensor = preprocess_image_vit(img_path)  # Preprocesar la imagen
    save_tensor_compressed(
        img_tensor, preprocessed_img_path
    )  # Guardar el tensor preprocesado


# Función para preprocesar imágenes en paralelo
def preprocess_images(data_dir, preprocessed_dir):
    if not os.path.exists(preprocessed_dir):
        os.makedirs(
            preprocessed_dir
        )  # Crear directorio para imágenes preprocesadas si no existe

    already_processed_all = True
    for emotion in os.listdir(data_dir):  # Iterar sobre las carpetas de emociones
        emotion_dir = os.path.join(data_dir, emotion)
        preprocessed_emotion_dir = os.path.join(preprocessed_dir, emotion)
        if not os.path.exists(preprocessed_emotion_dir):
            os.makedirs(
                preprocessed_emotion_dir
            )  # Crear directorio para la emoción preprocesada si no existe

        img_paths = []
        preprocessed_img_paths = []

        for img_name in os.listdir(
            emotion_dir
        ):  # Iterar sobre las imágenes en la carpeta de emociones
            img_path = os.path.join(emotion_dir, img_name)
            preprocessed_img_path = os.path.join(
                preprocessed_emotion_dir, img_name.split(".")[0] + ".pt.gz"
            )
            if not os.path.exists(preprocessed_img_path):
                img_paths.append(
                    img_path
                )  # Añadir a la lista de imágenes a preprocesar
                preprocessed_img_paths.append(
                    preprocessed_img_path
                )  # Añadir a la lista de rutas de imágenes preprocesadas

        if img_paths:
            already_processed_all = False
            with ThreadPoolExecutor() as executor:
                list(
                    tqdm(
                        executor.map(
                            preprocess_and_save_image, img_paths, preprocessed_img_paths
                        ),
                        total=len(img_paths),
                        desc=f"Preprocessing {emotion}",
                        ncols=80,
                    )
                )

        if not img_paths:
            print(f"Todas las imágenes de {emotion} ya han sido preprocesadas.")

    if already_processed_all:
        print("Todas las imágenes ya han sido preprocesadas.")
    return already_processed_all


# Clase personalizada para cargar imágenes preprocesadas
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []

        for emotion in os.listdir(data_dir):  # Iterar sobre las carpetas de emociones
            emotion_dir = os.path.join(data_dir, emotion)
            for img_name in os.listdir(
                emotion_dir
            ):  # Iterar sobre las imágenes en la carpeta de emociones
                self.data.append(
                    (os.path.join(emotion_dir, img_name), emotion)
                )  # Guardar la ruta de la imagen y su etiqueta

    def __len__(self):
        return len(self.data)  # Devolver la cantidad de datos

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            img = load_tensor_compressed(img_path)  # Cargar el tensor comprimido
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None, None

        label = torch.tensor(emotion_map[label])  # Convertir etiqueta a tensor
        if self.transform:
            img = self.transform(img)  # Aplicar transformaciones si existen
        return img, label  # Devolver imagen y etiqueta


# Función para cargar imágenes preprocesadas
def load_preprocessed_images(preprocessed_dir):
    return CustomDataset(preprocessed_dir)


# Función para preprocesar las imágenes de prueba en memoria usando ViT
def preprocess_test_images_vit(test_dir):
    test_images = []
    test_labels = []

    for emotion in os.listdir(test_dir):  # Iterar sobre las carpetas de emociones
        emotion_dir = os.path.join(test_dir, emotion)
        for img_name in tqdm(
            os.listdir(emotion_dir), desc=f"Preprocessing {emotion} (test)", ncols=80
        ):
            img_path = os.path.join(emotion_dir, img_name)
            img_tensor = preprocess_image_vit(img_path)  # Preprocesar la imagen
            test_images.append(
                img_tensor
            )  # Añadir tensor de imagen preprocesada a la lista
            test_labels.append(emotion_map[emotion])  # Añadir etiqueta a la lista

    return test_images, test_labels


# Función para entrenar el modelo
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=25):
    best_model_wts = copy.deepcopy(
        model.state_dict()
    )  # Guardar los mejores pesos del modelo
    best_acc = 0.0  # Inicializar mejor precisión
    scaler = GradScaler()  # Escalador para entrenamiento de precisión mixta
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Poner el modelo en modo de entrenamiento
            else:
                model.eval()  # Poner el modelo en modo de evaluación

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(
                dataloaders[phase], desc=f"{phase} phase", ncols=80
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    with autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs.logits, 1)
                        loss = criterion(outputs.logits, labels)

                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            # Guardar el modelo si es el mejor hasta ahora
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(
                    best_model_wts,
                    os.path.join(model_dir, f"best_model_epoch_{epoch}.pth"),
                )
                print(f"Mejor precisión encontrada: {best_acc:.4f}. Modelo guardado.")

        # Guardar estadísticas de cada época
        with open(os.path.join(model_dir, "training_history.txt"), "a") as f:
            f.write(f"Epoch {epoch}/{num_epochs - 1}\n")
            f.write(
                f"Train Loss: {history['train_loss'][-1]:.4f} Acc: {history['train_acc'][-1]:.4f}\n"
            )
            f.write(
                f"Val Loss: {history['val_loss'][-1]:.4f} Acc: {history['val_acc'][-1]:.4f}\n"
            )
            f.write("-" * 10 + "\n")

    model.load_state_dict(best_model_wts)
    return model, best_acc, history["train_acc"]


# Función para evaluar el modelo
def evaluate_model_vit(model, test_images, test_labels):
    model.eval()  # Poner el modelo en modo de evaluación
    y_true = []
    y_pred = []

    for img_tensor, label in tqdm(
        zip(test_images, test_labels),
        desc="Evaluating",
        total=len(test_images),
        ncols=80,
    ):
        inputs = img_tensor.unsqueeze(0).to(
            device
        )  # Añadir dimensión de lote y mover a dispositivo
        labels = torch.tensor([label]).to(
            device
        )  # Convertir etiqueta a tensor y mover a dispositivo

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs.logits, 1)

        y_true.append(labels.item())  # Añadir etiqueta verdadera a la lista
        y_pred.append(preds.item())  # Añadir predicción a la lista

    return y_true, y_pred


if __name__ == "__main__":
    # Verificar si las imágenes ya están preprocesadas
    already_processed_all = preprocess_images(
        os.path.join(data_dir, "train"), preprocessed_dir
    )

    # Verificar si el modelo ya está entrenado
    model_path = os.path.join(model_dir, "best_model.pth")
    train_accuracies = []
    if os.path.exists(model_path):
        print("Cargando el modelo entrenado...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        if already_processed_all:
            # Evaluar el modelo cargado y mostrar la precisión
            test_images, test_labels = preprocess_test_images_vit(
                os.path.join(data_dir, "test")
            )
            y_true, y_pred = evaluate_model_vit(model, test_images, test_labels)
            total_correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
            total_samples = len(y_true)
            total_accuracy = total_correct / total_samples * 100
            print(f"Acc: {total_accuracy:.2f}%")
    else:
        print("Entrenando el modelo...")
        # Cargar el conjunto de datos preprocesado
        train_dataset = load_preprocessed_images(preprocessed_dir)
        dataloaders = {
            "train": DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
            ),
            "val": DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
            ),  # Usar el mismo conjunto para validación
        }
        dataset_sizes = {x: len(dataloaders[x].dataset) for x in ["train", "val"]}

        # Configurar el modelo
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reducido

        # Crear la carpeta AI si no existe
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Entrenar el modelo
        model, best_acc, train_accuracies = train_model(
            model,
            criterion,
            optimizer,
            dataloaders,
            dataset_sizes,
            num_epochs=num_epochs,
        )

        # Guardar el modelo
        torch.save(model.state_dict(), model_path)
        print(f"Modelo guardado con precisión: {best_acc:.2f}")

    # Preprocesar las imágenes de prueba en memoria
    test_images, test_labels = preprocess_test_images_vit(
        os.path.join(data_dir, "test")
    )

    # Evaluar el modelo
    y_true, y_pred = evaluate_model_vit(model, test_images, test_labels)

    # Calcular precisión por clase y precisión total
    precision_per_class = {}
    total_correct = 0
    total_samples = len(y_true)

    for label in range(7):
        true_positives = sum(
            (y_true[i] == label) and (y_pred[i] == label) for i in range(len(y_true))
        )
        total = sum(y_true[i] == label for i in range(len(y_true)))
        precision_per_class[label] = true_positives / total * 100 if total > 0 else 0
        total_correct += true_positives

    total_accuracy = total_correct / total_samples * 100

    # Imprimir precisión por clase y precisión total
    for label, precision in precision_per_class.items():
        emotion = list(emotion_map.keys())[label]
        print(f"{emotion} obtuvo un {precision:.2f}% de acierto")

    print(f"Precisión total del modelo: {total_accuracy:.2f}%")

    # Calcular y mostrar el promedio de precisión durante el entrenamiento
    if train_accuracies:
        average_train_accuracy = sum(train_accuracies) / len(train_accuracies) * 100
        print(
            f"Promedio de precisión durante el entrenamiento: {average_train_accuracy:.2f}%"
        )
    else:
        print(f"Acc: {total_accuracy:.2f}%")