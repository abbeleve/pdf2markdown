const dropArea = document.getElementById('dropArea');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const form = document.getElementById('uploadForm');
const submitBtn = document.getElementById('submitBtn');
const resultDiv = document.getElementById('result');
const dropText = document.getElementById('dropText');
const fileNameDiv = document.getElementById('fileName');

let currentFile = null;

// Открыть выбор файла по кнопке
browseBtn.addEventListener('click', () => fileInput.click());

// Обновить отображение имени файла
function updateFileDisplay(file) {
  currentFile = file;
  if (file) {
    dropText.textContent = 'Файл выбран:';
    fileNameDiv.textContent = file.name;
    submitBtn.disabled = false;
    dropArea.classList.add('file-selected');
  } else {
    dropText.textContent = 'Перетащите PDF сюда';
    fileNameDiv.textContent = '';
    submitBtn.disabled = true;
    dropArea.classList.remove('file-selected');
  }
}

// Drag & drop
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
  dropArea.addEventListener(event, preventDefaults, false);
});

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

['dragenter', 'dragover'].forEach(event => {
  dropArea.addEventListener(event, () => {
    if (!currentFile) dropArea.classList.add('drag-over');
  }, false);
});

['dragleave', 'drop'].forEach(event => {
  dropArea.addEventListener(event, () => {
    dropArea.classList.remove('drag-over');
  }, false);
});

dropArea.addEventListener('drop', (e) => {
  const files = e.dataTransfer.files;
  if (files.length && files[0].name.endsWith('.pdf')) {
    fileInput.files = files;
    updateFileDisplay(files[0]);
  } else {
    showError('Пожалуйста, загрузите PDF-файл.');
  }
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) {
    if (!file.name.endsWith('.pdf')) {
      showError('Поддерживаются только PDF-файлы.');
      fileInput.value = '';
      updateFileDisplay(null);
    } else {
      updateFileDisplay(file);
    }
  } else {
    updateFileDisplay(null);
  }
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!currentFile) {
    showError('Выберите файл.');
    return;
  }

  const formData = new FormData();
  formData.append('file', currentFile);

  submitBtn.disabled = true;
  submitBtn.innerHTML = '<span class="spinner"></span> Обработка...';
  resultDiv.innerHTML = '';

  try {
    const response = await fetch('/convert', {
      method: 'POST',
      body: formData
    });

    if (response.ok) {
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      const filename = getFilenameFromResponse(response) || 'output.md';
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      showSuccess(`✅ ${filename} готов!`);
      // Опционально: сбросить выбор после скачивания
      // updateFileDisplay(null);
    } else {
      const errorText = await response.text();
      showError(`❌ ${errorText}`);
    }
  } catch (err) {
    showError(`❌ Ошибка: ${err.message}`);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Конвертировать';
  }
});

function getFilenameFromResponse(response) {
  const disposition = response.headers.get('content-disposition');
  if (disposition && disposition.includes('filename=')) {
    return disposition.split('filename=')[1].replace(/"/g, '');
  }
  return null;
}

function showSuccess(message) {
  resultDiv.className = 'result success';
  resultDiv.textContent = message;
}

function showError(message) {
  resultDiv.className = 'result error';
  resultDiv.textContent = message;
}