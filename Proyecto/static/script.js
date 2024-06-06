document.getElementById('file-input').addEventListener('change', function (event) {
    const fileInput = event.target;
    const messagePlaceholder = document.getElementById('message-placeholder');

    if (fileInput.files.length > 0) {
        messagePlaceholder.textContent = "Imagen cargada";
    } else {
        messagePlaceholder.textContent = "Cargue una imagen";
    }
});

document.getElementById('upload-form').addEventListener('submit', async function (event) {
    event.preventDefault();

    const fileInput = document.getElementById('file-input');
    const messagePlaceholder = document.getElementById('message-placeholder');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const loading = document.getElementById('loading');

    if (fileInput.files.length === 0) {
        showNotification("No ha cargado una imagen", "error");
        return;
    }

    messagePlaceholder.style.display = "none";
    loading.style.display = 'flex';

    const chatBox = document.getElementById('chat-box');
    const sentimentHistory = document.getElementById('sentiment-history');

    const userMessage = document.createElement('div');
    userMessage.className = 'message user';
    userMessage.innerHTML = '<div class="text">Usuario: ha mandado una imagen</div>';
    chatBox.appendChild(userMessage);

    const progressInterval = setInterval(() => {
        if (parseInt(progressBarFill.style.width) >= 100) {
            clearInterval(progressInterval);
        } else {
            progressBarFill.style.width = `${parseInt(progressBarFill.style.width) + 10}%`;
        }
    }, 500);

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        clearInterval(progressInterval);
        progressBarFill.style.width = "100%";
        setTimeout(() => {
            loading.style.display = 'none';
            progressBarFill.style.width = "0%";

            const userImageMessage = document.createElement('div');
            userImageMessage.className = 'message user';
            const userImage = document.createElement('img');
            userImage.src = URL.createObjectURL(fileInput.files[0]);
            userImageMessage.appendChild(userImage);
            chatBox.appendChild(userImageMessage);

            const botMessage = document.createElement('div');
            botMessage.className = 'message bot';
            botMessage.innerHTML = `<div class="text">Bot: ${result.sentiment ? 'El sentimiento detectado es ' + result.sentiment : 'No se detectó ningún rostro en la imagen.'}</div>`;
            chatBox.appendChild(botMessage);

            const historyItem = document.createElement('li');
            historyItem.textContent = `Sentimiento: ${result.sentiment || 'No se detectó ningún rostro'}`;
            sentimentHistory.appendChild(historyItem);

            chatBox.scrollTop = chatBox.scrollHeight;

            fileInput.value = '';
            messagePlaceholder.style.display = "block";
            messagePlaceholder.textContent = "Cargue una imagen";

            showNotification("Predicción completada con éxito", "success");
        }, 1000);
    } catch (error) {
        clearInterval(progressInterval);
        loading.style.display = 'none';
        progressBarFill.style.width = "0%";
        showNotification("Error al predecir la emoción: " + error.message, "error");
    }
});

document.addEventListener('paste', function (event) {
    const items = event.clipboardData.items;
    for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
            const file = items[i].getAsFile();
            const fileInput = document.getElementById('file-input');
            const messagePlaceholder = document.getElementById('message-placeholder');

            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;

            messagePlaceholder.textContent = "Imagen cargada";
        }
    }
});

// Función para mostrar notificaciones
function showNotification(message, type) {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
            notification.remove();
        }, 500);
    }, 3000);
}

