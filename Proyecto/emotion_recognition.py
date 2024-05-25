import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

# Configuraciones
data_dir = "./archive"
preprocessed_dir = "./preprocessed_images"
model_dir = "./AI"
batch_size = 32
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms para preprocesar las imágenes
data_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Diccionario para mapear etiquetas
emotion_map = {
    "angry": 0,
    "disgusted": 1,
    "fearful": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprised": 6,
}


# Función para preprocesar imágenes y guardarlas como tensores .pt
def preprocess_images(data_dir, preprocessed_dir):
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)

    for emotion in os.listdir(data_dir):
        emotion_dir = os.path.join(data_dir, emotion)
        preprocessed_emotion_dir = os.path.join(preprocessed_dir, emotion)
        if not os.path.exists(preprocessed_emotion_dir):
            os.makedirs(preprocessed_emotion_dir)

        already_processed = True
        for img_name in tqdm(
            os.listdir(emotion_dir), desc=f"Preprocessing {emotion}", ncols=80
        ):
            img_path = os.path.join(emotion_dir, img_name)
            preprocessed_img_path = os.path.join(
                preprocessed_emotion_dir, img_name.split(".")[0] + ".pt"
            )
            if os.path.exists(preprocessed_img_path):
                continue

            img = Image.open(img_path)
            img_tensor = data_transforms(img)
            torch.save(img_tensor, preprocessed_img_path)
            already_processed = False

        if already_processed:
            print(f"Todas las imágenes de {emotion} ya han sido preprocesadas.")
        else:
            print(f"Todas las imágenes de {emotion} han sido preprocesadas.")


# Función para cargar imágenes preprocesadas
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []

        for emotion in os.listdir(data_dir):
            emotion_dir = os.path.join(data_dir, emotion)
            for img_name in os.listdir(emotion_dir):
                self.data.append((os.path.join(emotion_dir, img_name), emotion))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            img = torch.load(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None, None

        label = torch.tensor(emotion_map[label])
        if self.transform:
            img = self.transform(img)
        return img, label


def load_preprocessed_images(preprocessed_dir):
    return CustomDataset(preprocessed_dir)


# Función para preprocesar las imágenes de prueba en memoria
def preprocess_test_images(test_dir):
    test_images = []
    test_labels = []

    for emotion in os.listdir(test_dir):
        emotion_dir = os.path.join(test_dir, emotion)
        for img_name in tqdm(
            os.listdir(emotion_dir), desc=f"Preprocessing {emotion} (test)", ncols=80
        ):
            img_path = os.path.join(emotion_dir, img_name)
            img = Image.open(img_path)
            img_tensor = data_transforms(img)
            test_images.append(img_tensor)
            test_labels.append(emotion_map[emotion])

    return test_images, test_labels


# Función para entrenar el modelo
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(
                dataloaders[phase], desc=f"{phase} phase", ncols=80
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

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
    return model, best_acc


# Función para evaluar el modelo
def evaluate_model(model, test_images, test_labels):
    model.eval()
    y_true = []
    y_pred = []

    for img_tensor, label in tqdm(
        zip(test_images, test_labels), desc="Evaluating", ncols=80
    ):
        inputs = img_tensor.unsqueeze(0).to(device)
        labels = torch.tensor([label]).to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        y_true.append(labels.item())
        y_pred.append(preds.item())

    return y_true, y_pred


# Verificar si las imágenes ya están preprocesadas
if not os.path.exists(preprocessed_dir):
    preprocess_images(os.path.join(data_dir, "train"), preprocessed_dir)

# Verificar si el modelo ya está entrenado
model_path = os.path.join(model_dir, "best_model.pth")
if os.path.exists(model_path):
    print("Cargando el modelo entrenado...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
else:
    print("Entrenando el modelo...")
    # Cargar el conjunto de datos preprocesado
    train_dataset = load_preprocessed_images(preprocessed_dir)
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        ),  # Usar el mismo conjunto para validación
    }
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ["train", "val"]}

    # Configurar el modelo
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 7)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Crear la carpeta AI si no existe
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Entrenar el modelo
    model, best_acc = train_model(
        model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=num_epochs
    )

    # Guardar el modelo
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado con precisión: {best_acc:.2f}")

# Preprocesar las imágenes de prueba en memoria
test_images, test_labels = preprocess_test_images(os.path.join(data_dir, "test"))

# Evaluar el modelo
y_true, y_pred = evaluate_model(model, test_images, test_labels)

# Calcular precisión por clase
precision_per_class = {}
for label in range(7):
    true_positives = sum(
        (y_true[i] == label) and (y_pred[i] == label) for i in range(len(y_true))
    )
    total = sum(y_true[i] == label for i in range(len(y_true)))
    precision_per_class[label] = true_positives / total * 100 if total > 0 else 0

for emotion, precision in precision_per_class.items():
    print(f"{emotion} obtuvo un {precision:.2f}% de acierto")
