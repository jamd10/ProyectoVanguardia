// Añadir un evento al input de archivo para cambiar el mensaje de placeholder cuando se seleccione una imagen
document.getElementById('file-input').addEventListener('change', function (event) {
    const fileInput = event.target;
    const messagePlaceholder = document.getElementById('message-placeholder');

    if (fileInput.files.length > 0) {
        messagePlaceholder.textContent = "Imagen cargada";  // Cambiar mensaje cuando una imagen está cargada
    } else {
        messagePlaceholder.textContent = "Cargue una imagen";  // Restaurar mensaje si no hay imagen cargada
    }
});

// Añadir un evento al formulario para manejar la subida de la imagen
document.getElementById('upload-form').addEventListener('submit', async function (event) {
    event.preventDefault();  // Prevenir el comportamiento por defecto del formulario

    const fileInput = document.getElementById('file-input');
    const messagePlaceholder = document.getElementById('message-placeholder');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const loading = document.getElementById('loading');

    if (fileInput.files.length === 0) {
        messagePlaceholder.textContent = "No ha cargado una imagen";  // Mostrar mensaje si no se ha cargado ninguna imagen
        return;
    }

    messagePlaceholder.style.display = "none";  // Ocultar el mensaje de placeholder
    loading.style.display = 'flex';  // Mostrar la barra de carga

    const chatBox = document.getElementById('chat-box');
    const sentimentHistory = document.getElementById('sentiment-history');

    // Crear un mensaje en el chat indicando que el usuario ha mandado una imagen
    const userMessage = document.createElement('div');
    userMessage.className = 'message user';
    userMessage.innerHTML = '<div class="text">Usuario: ha mandado una imagen</div>';
    chatBox.appendChild(userMessage);

    // Simulación de progreso de la barra de carga
    const progressInterval = setInterval(() => {
        if (progressBarFill.style.width === "100%") {
            clearInterval(progressInterval);
        } else {
            progressBarFill.style.width = `${parseInt(progressBarFill.style.width) + 10}%`;
        }
    }, 500);

    // Crear un objeto FormData y añadir el archivo de imagen
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    // Enviar la imagen al servidor para predecir el sentimiento
    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    // Obtener la respuesta del servidor
    const result = await response.json();

    // Detener la barra de progreso y completar la animación de carga
    clearInterval(progressInterval);
    progressBarFill.style.width = "100%";
    setTimeout(() => {
        loading.style.display = 'none';
        progressBarFill.style.width = "0%";

        // Mostrar la imagen cargada en el chat
        const userImageMessage = document.createElement('div');
        userImageMessage.className = 'message user';
        const userImage = document.createElement('img');
        userImage.src = URL.createObjectURL(fileInput.files[0]);
        userImageMessage.appendChild(userImage);
        chatBox.appendChild(userImageMessage);

        // Mostrar el mensaje del bot con el sentimiento detectado
        const botMessage = document.createElement('div');
        botMessage.className = 'message bot';
        if (result.sentiment) {
            botMessage.innerHTML = `<div class="text">Bot: El sentimiento detectado es ${result.sentiment}</div>`;
        } else {
            botMessage.innerHTML = `<div class="text">Bot: No se detectó ningún rostro en la imagen.</div>`;
        }
        chatBox.appendChild(botMessage);

        // Añadir el resultado al historial de sentimientos
        const historyItem = document.createElement('li');
        if (result.sentiment) {
            historyItem.textContent = `Sentimiento: ${result.sentiment}`;
        } else {
            historyItem.textContent = `Sentimiento: No se detectó ningún rostro`;
        }
        sentimentHistory.appendChild(historyItem);

        // Desplazar el chat hacia abajo para mostrar el nuevo mensaje
        chatBox.scrollTop = chatBox.scrollHeight;

        // Limpiar el input de archivo y restaurar el placeholder
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

            messagePlaceholder.textContent = "Imagen cargada";  // Cambiar mensaje cuando una imagen está pegada
        }
    }
});
