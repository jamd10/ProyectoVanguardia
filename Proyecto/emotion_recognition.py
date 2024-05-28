import os  # Manejo de operaciones del sistema de archivos
import copy  # Para hacer copias profundas de objetos
import torch  # Biblioteca principal para computación de tensores y redes neuronales
import torch.nn as nn  # Módulo para construir y entrenar modelos de redes neuronales
import torch.optim as optim  # Módulo para algoritmos de optimización
from transformers import ViTForImageClassification, ViTImageProcessor  # Modelos y procesadores de imágenes preentrenados de Hugging Face
from torch.utils.data import DataLoader, Dataset  # Para manejar datos de entrenamiento y prueba
from tqdm import tqdm  # Mostrar barras de progreso
from PIL import Image  # Manejo de operaciones de imagen
from torch.cuda.amp import autocast, GradScaler  # Entrenamiento de precisión mixta
import gzip  # Compresión y descompresión de archivos
from concurrent.futures import ThreadPoolExecutor  # Manejo de concurrencia
import warnings  # Manejo de warnings

# Suprimir warnings específicos
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

# Configuraciones
data_dir = "./archive"  # Directorio donde se encuentran los datos originales
preprocessed_dir = "./preprocessed_images"  # Directorio para almacenar imágenes preprocesadas
model_dir = "./AI"  # Directorio para guardar el modelo entrenado
batch_size = 64  # Tamaño del lote para entrenamiento
num_epochs = 10  # Número de épocas para el entrenamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Usar GPU si está disponible, de lo contrario usar CPU
model_name = "google/vit-base-patch16-224"  # Nombre del modelo preentrenado de Vision Transformer (ViT)

# Cargar el modelo y el feature extractor de Hugging Face
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=7,
    ignore_mismatched_sizes=True,  # Cargar el modelo con 7 etiquetas y permitir tamaños desajustados
)
model.to(device)  # Mover el modelo al dispositivo (GPU o CPU)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)  # Cargar el procesador de imágenes

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
    inputs = feature_extractor(images=image, return_tensors="pt")  # Extraer características y convertir a tensores de PyTorch
    return inputs["pixel_values"].squeeze(0).half()  # Convertir a float16 para reducir el uso de memoria

# Función para guardar tensores comprimidos
def save_tensor_compressed(tensor, filepath):
    with gzip.GzipFile(filepath, "wb") as f:  # Abrir archivo comprimido para escritura
        torch.save(tensor, f, pickle_protocol=4)  # Guardar tensor con protocolo de pickle 4

# Función para cargar tensores comprimidos
def load_tensor_compressed(filepath):
    with gzip.GzipFile(filepath, "rb") as f:  # Abrir archivo comprimido para lectura
        return torch.load(f)  # Cargar tensor desde el archivo

# Función para preprocesar y guardar una sola imagen
def preprocess_and_save_image(img_path, preprocessed_img_path):
    img_tensor = preprocess_image_vit(img_path)  # Preprocesar la imagen
    save_tensor_compressed(img_tensor, preprocessed_img_path)  # Guardar el tensor preprocesado

# Función para preprocesar imágenes en paralelo
def preprocess_images(data_dir, preprocessed_dir):
    if not os.path.exists(preprocessed_dir):  # Si el directorio no existe
        os.makedirs(preprocessed_dir)  # Crear directorio para imágenes preprocesadas

    already_processed_all = True  # Bandera para verificar si todas las imágenes ya están preprocesadas
    for emotion in os.listdir(data_dir):  # Iterar sobre las carpetas de emociones
        emotion_dir = os.path.join(data_dir, emotion)  # Ruta de la carpeta de la emoción
        preprocessed_emotion_dir = os.path.join(preprocessed_dir, emotion)  # Ruta de la carpeta de la emoción preprocesada
        if not os.path.exists(preprocessed_emotion_dir):  # Si el directorio no existe
            os.makedirs(preprocessed_emotion_dir)  # Crear directorio para la emoción preprocesada

        img_paths = []  # Lista para almacenar rutas de imágenes
        preprocessed_img_paths = []  # Lista para almacenar rutas de imágenes preprocesadas

        for img_name in os.listdir(emotion_dir):  # Iterar sobre las imágenes en la carpeta de emociones
            img_path = os.path.join(emotion_dir, img_name)  # Ruta de la imagen
            preprocessed_img_path = os.path.join(preprocessed_emotion_dir, img_name.split(".")[0] + ".pt.gz")  # Ruta de la imagen preprocesada
            if not os.path.exists(preprocessed_img_path):  # Si la imagen no ha sido preprocesada
                img_paths.append(img_path)  # Añadir a la lista de imágenes a preprocesar
                preprocessed_img_paths.append(preprocessed_img_path)  # Añadir a la lista de rutas de imágenes preprocesadas

        if img_paths:  # Si hay imágenes a preprocesar
            already_processed_all = False  # Cambiar la bandera
            with ThreadPoolExecutor() as executor:  # Ejecutar en paralelo
                list(tqdm(
                    executor.map(preprocess_and_save_image, img_paths, preprocessed_img_paths),  # Preprocesar y guardar imágenes
                    total=len(img_paths),  # Total de imágenes
                    desc=f"Preprocessing {emotion}",  # Descripción de la barra de progreso
                    ncols=80,  # Ancho de la barra de progreso
                ))

        if not img_paths:  # Si no hay imágenes a preprocesar
            print(f"Todas las imágenes de {emotion} ya han sido preprocesadas.")  # Imprimir mensaje

    if already_processed_all:  # Si todas las imágenes ya estaban preprocesadas
        print("Todas las imágenes ya han sido preprocesadas.")  # Imprimir mensaje
    return already_processed_all  # Devolver el estado de preprocesamiento

# Clase personalizada para cargar imágenes preprocesadas
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):  # Inicialización de la clase
        self.data_dir = data_dir  # Directorio de datos
        self.transform = transform  # Transformaciones a aplicar
        self.data = []  # Lista para almacenar datos

        for emotion in os.listdir(data_dir):  # Iterar sobre las carpetas de emociones
            emotion_dir = os.path.join(data_dir, emotion)  # Ruta de la carpeta de la emoción
            for img_name in os.listdir(emotion_dir):  # Iterar sobre las imágenes en la carpeta de emociones
                self.data.append((os.path.join(emotion_dir, img_name), emotion))  # Guardar la ruta de la imagen y su etiqueta

    def __len__(self):  # Devolver la cantidad de datos
        return len(self.data)  # Longitud de la lista de datos

    def __getitem__(self, idx):  # Obtener un elemento por índice
        img_path, label = self.data[idx]  # Obtener ruta de la imagen y etiqueta
        try:
            img = load_tensor_compressed(img_path)  # Cargar el tensor comprimido
        except Exception as e:
            print(f"Error loading {img_path}: {e}")  # Imprimir error si no se puede cargar
            return None, None  # Devolver valores nulos

        label = torch.tensor(emotion_map[label])  # Convertir etiqueta a tensor
        if self.transform:  # Si hay transformaciones
            img = self.transform(img)  # Aplicar transformaciones
        return img, label  # Devolver imagen y etiqueta

# Función para cargar imágenes preprocesadas
def load_preprocessed_images(preprocessed_dir):
    return CustomDataset(preprocessed_dir)  # Devolver dataset personalizado

# Función para preprocesar las imágenes de prueba en memoria usando ViT
def preprocess_test_images_vit(test_dir):
    test_images = []  # Lista para almacenar imágenes de prueba
    test_labels = []  # Lista para almacenar etiquetas de prueba

    for emotion in os.listdir(test_dir):  # Iterar sobre las carpetas de emociones
        emotion_dir = os.path.join(test_dir, emotion)  # Ruta de la carpeta de la emoción
        for img_name in tqdm(os.listdir(emotion_dir), desc=f"Preprocessing {emotion} (test)", ncols=80):  # Iterar sobre las imágenes en la carpeta de emociones
            img_path = os.path.join(emotion_dir, img_name)  # Ruta de la imagen
            img_tensor = preprocess_image_vit(img_path)  # Preprocesar la imagen
            test_images.append(img_tensor)  # Añadir tensor de imagen preprocesada a la lista
            test_labels.append(emotion_map[emotion])  # Añadir etiqueta a la lista

    return test_images, test_labels  # Devolver imágenes y etiquetas preprocesadas

# Función para entrenar el modelo
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())  # Guardar los mejores pesos del modelo
    best_acc = 0.0  # Inicializar mejor precisión
    scaler = GradScaler()  # Escalador para entrenamiento de precisión mixta
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}  # Diccionario para almacenar la historia de entrenamiento

    for epoch in range(num_epochs):  # Iterar sobre el número de épocas
        print(f"Epoch {epoch}/{num_epochs - 1}")  # Imprimir número de época
        print("-" * 10)  # Imprimir separación

        for phase in ["train", "val"]:  # Iterar sobre las fases de entrenamiento y validación
            if phase == "train":  # Si es la fase de entrenamiento
                model.train()  # Poner el modelo en modo de entrenamiento
            else:  # Si es la fase de validación
                model.eval()  # Poner el modelo en modo de evaluación

            running_loss = 0.0  # Inicializar pérdida acumulada
            running_corrects = 0  # Inicializar aciertos acumulados

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} phase", ncols=80):  # Iterar sobre los datos de la fase
                inputs = inputs.to(device)  # Mover datos al dispositivo
                labels = labels.to(device)  # Mover etiquetas al dispositivo

                optimizer.zero_grad()  # Reiniciar gradientes

                with torch.set_grad_enabled(phase == "train"):  # Habilitar gradientes solo en entrenamiento
                    with autocast():  # Usar autocast para precisión mixta
                        outputs = model(inputs)  # Pasar datos por el modelo
                        _, preds = torch.max(outputs.logits, 1)  # Obtener predicciones
                        loss = criterion(outputs.logits, labels)  # Calcular pérdida

                    if phase == "train":  # Si es la fase de entrenamiento
                        scaler.scale(loss).backward()  # Hacer backward con escala
                        scaler.step(optimizer)  # Actualizar parámetros del optimizador
                        scaler.update()  # Actualizar escala

                running_loss += loss.item() * inputs.size(0)  # Acumular pérdida
                running_corrects += torch.sum(preds == labels.data)  # Acumular aciertos

            epoch_loss = running_loss / dataset_sizes[phase]  # Calcular pérdida de la época
            epoch_acc = running_corrects.double() / dataset_sizes[phase]  # Calcular precisión de la época

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")  # Imprimir resultados de la época

            history[f"{phase}_loss"].append(epoch_loss)  # Guardar pérdida de la época en la historia
            history[f"{phase}_acc"].append(epoch_acc)  # Guardar precisión de la época en la historia

            if phase == "val" and epoch_acc > best_acc:  # Si es la fase de validación y la precisión es la mejor
                best_acc = epoch_acc  # Actualizar mejor precisión
                best_model_wts = copy.deepcopy(model.state_dict())  # Guardar los mejores pesos del modelo
                torch.save(best_model_wts, os.path.join(model_dir, f"best_model_epoch_{epoch}.pth"))  # Guardar el modelo
                print(f"Mejor precisión encontrada: {best_acc:.4f}. Modelo guardado.")  # Imprimir mensaje

        with open(os.path.join(model_dir, "training_history.txt"), "a") as f:  # Abrir archivo de historia de entrenamiento
            f.write(f"Epoch {epoch}/{num_epochs - 1}\n")  # Escribir número de época
            f.write(f"Train Loss: {history['train_loss'][-1]:.4f} Acc: {history['train_acc'][-1]:.4f}\n")  # Escribir resultados de entrenamiento
            f.write(f"Val Loss: {history['val_loss'][-1]:.4f} Acc: {history['val_acc'][-1]:.4f}\n")  # Escribir resultados de validación
            f.write("-" * 10 + "\n")  # Escribir separación

    model.load_state_dict(best_model_wts)  # Cargar los mejores pesos del modelo
    return model, best_acc, history["train_acc"]  # Devolver modelo, mejor precisión y historia de entrenamiento

# Función para evaluar el modelo
def evaluate_model_vit(model, test_images, test_labels):
    model.eval()  # Poner el modelo en modo de evaluación
    y_true = []  # Lista para almacenar etiquetas verdaderas
    y_pred = []  # Lista para almacenar predicciones

    for img_tensor, label in tqdm(zip(test_images, test_labels), desc="Evaluating", total=len(test_images), ncols=80):  # Iterar sobre imágenes y etiquetas de prueba
        inputs = img_tensor.unsqueeze(0).to(device)  # Añadir dimensión de lote y mover a dispositivo
        labels = torch.tensor([label]).to(device)  # Convertir etiqueta a tensor y mover a dispositivo

        with torch.no_grad():  # Deshabilitar gradientes
            outputs = model(inputs)  # Pasar datos por el modelo
            _, preds = torch.max(outputs.logits, 1)  # Obtener predicciones

        y_true.append(labels.item())  # Añadir etiqueta verdadera a la lista
        y_pred.append(preds.item())  # Añadir predicción a la lista

    return y_true, y_pred  # Devolver etiquetas verdaderas y predicciones

if __name__ == "__main__":
    already_processed_all = preprocess_images(os.path.join(data_dir, "train"), preprocessed_dir)  # Verificar si las imágenes ya están preprocesadas

    model_path = os.path.join(model_dir, "best_model.pth")  # Ruta del modelo entrenado
    history_path = os.path.join(model_dir, "training_history.txt")  # Ruta del archivo de historia de entrenamiento
    train_accuracies = []  # Lista para almacenar precisiones de entrenamiento

    if os.path.exists(model_path):  # Si el modelo ya está entrenado
        if os.path.exists(history_path):  # Si existe el archivo de historia de entrenamiento
            print("Historial de entrenamiento:")  # Imprimir mensaje
            with open(history_path, "r") as f:  # Abrir archivo de historia de entrenamiento
                print(f.read())  # Imprimir contenido del archivo

        print("Cargando el modelo entrenado...")  # Imprimir mensaje
        model.load_state_dict(torch.load(model_path, map_location=device))  # Cargar el modelo entrenado
        model.to(device)  # Mover el modelo al dispositivo
    else:  # Si el modelo no está entrenado
        print("Entrenando el modelo...")  # Imprimir mensaje
        train_dataset = load_preprocessed_images(preprocessed_dir)  # Cargar el conjunto de datos preprocesado
        dataloaders = {
            "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),  # Cargar datos de entrenamiento
            "val": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),  # Cargar datos de validación
        }
        dataset_sizes = {x: len(dataloaders[x].dataset) for x in ["train", "val"]}  # Calcular tamaños de conjuntos de datos

        criterion = nn.CrossEntropyLoss()  # Función de pérdida
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Optimizador

        if not os.path.exists(model_dir):  # Si el directorio no existe
            os.makedirs(model_dir)  # Crear el directorio

        model, best_acc, train_accuracies = train_model(
            model,
            criterion,
            optimizer,
            dataloaders,
            dataset_sizes,
            num_epochs=num_epochs,
        )  # Entrenar el modelo

        torch.save(model.state_dict(), model_path)  # Guardar el modelo
        print(f"Modelo guardado con precisión: {best_acc:.2f}")  # Imprimir mensaje

    test_images, test_labels = preprocess_test_images_vit(os.path.join(data_dir, "test"))  # Preprocesar las imágenes de prueba
    y_true, y_pred = evaluate_model_vit(model, test_images, test_labels)  # Evaluar el modelo

    precision_per_class = {}  # Diccionario para almacenar precisión por clase
    total_correct = 0  # Inicializar aciertos totales
    total_samples = len(y_true)  # Calcular total de muestras

    for label in range(7):  # Iterar sobre cada etiqueta
        true_positives = sum((y_true[i] == label) and (y_pred[i] == label) for i in range(len(y_true)))  # Calcular verdaderos positivos
        total = sum(y_true[i] == label for i in range(len(y_true)))  # Calcular total de muestras por etiqueta
        precision_per_class[label] = true_positives / total * 100 if total > 0 else 0  # Calcular precisión por clase
        total_correct += true_positives  # Acumular aciertos totales

    total_accuracy = total_correct / total_samples * 100  # Calcular precisión total

    for label, precision in precision_per_class.items():  # Iterar sobre la precisión por clase
        emotion = list(emotion_map.keys())[label]  # Obtener el nombre de la emoción
        print(f"{emotion} obtuvo un {precision:.2f}% de acierto")  # Imprimir precisión por clase

    print(f"Precisión total del modelo: {total_accuracy:.2f}%")  # Imprimir precisión total

    if train_accuracies:  # Si hay precisiones de entrenamiento
        average_train_accuracy = sum(train_accuracies) / len(train_accuracies) * 100  # Calcular promedio de precisión de entrenamiento
        print(f"Promedio de precisión durante el entrenamiento: {average_train_accuracy:.2f}%")  # Imprimir promedio de precisión
    else:
        print(f"Acc: {total_accuracy:.2f}%")  # Imprimir precisión total
