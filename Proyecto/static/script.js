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
        messagePlaceholder.textContent = "No ha cargado una imagen";
        return;
    }

    messagePlaceholder.style.display = "none";
    loading.style.display = 'flex';

    const chatBox = document.getElementById('chat-box');
    const sentimentHistory = document.getElementById('sentiment-history');

    const userMessage = document.createElement('div');
    userMessage.className = 'message user';
    userMessage.innerHTML = '<div class="text">Usuario: ha mandado un mensaje</div>';
    chatBox.appendChild(userMessage);

    const progressInterval = setInterval(() => {
        if (progressBarFill.style.width === "100%") {
            clearInterval(progressInterval);
        } else {
            progressBarFill.style.width = `${parseInt(progressBarFill.style.width) + 10}%`;
        }
    }, 500);

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

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
        botMessage.innerHTML = `<div class="text">Bot: El sentimiento detectado es ${result.sentiment}</div>`;
        chatBox.appendChild(botMessage);

        const historyItem = document.createElement('li');
        historyItem.textContent = `Sentimiento: ${result.sentiment}`;
        sentimentHistory.appendChild(historyItem);

        chatBox.scrollTop = chatBox.scrollHeight;

        // Limpiar input de archivo y vista previa
        fileInput.value = '';
        messagePlaceholder.style.display = "block";
        messagePlaceholder.textContent = "Cargue una imagen";
    }, 1000);
});

// Añadir evento para pegar imagen desde el portapapeles
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
