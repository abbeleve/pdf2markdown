import os
import uuid
import subprocess
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def main_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>PDF → Markdown</title>
      <style>
        body { font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; }
        input[type="file"] { margin: 10px 0; }
        button { padding: 8px 16px; font-size: 16px; }
        .result { margin-top: 20px; }
      </style>
    </head>
    <body>
      <h2>Перевод PDF в Markdown</h2>
      <form id="uploadForm" enctype="multipart/form-data">
        <input type="file" name="file" accept=".pdf" required />
        <button type="submit">Конвертировать</button>
      </form>
      <div class="result" id="result"></div>

      <script>
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
          e.preventDefault();
          const formData = new FormData(e.target);
          const resultDiv = document.getElementById('result');
          resultDiv.innerHTML = 'Обработка...';

          try {
            const response = await fetch('/convert', {
              method: 'POST',
              body: formData
            });

            if (response.ok) {
              const blob = await response.blob();
              const url = window.URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = 'output.md';
              a.click();
              window.URL.revokeObjectURL(url);
              resultDiv.innerHTML = '✅ Готово! Файл скачан.';
            } else {
              const err = await response.text();
              resultDiv.innerHTML = `❌ Ошибка: ${err}`;
            }
          } catch (err) {
            resultDiv.innerHTML = `❌ Ошибка: ${err.message}`;
          }
        });
      </script>
    </body>
    </html>
    """

@app.post("/convert")
async def convert_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Только PDF-файлы!")

    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}.md")

    # Сохраняем PDF
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Запускаем markitdown как CLI
    try:
        result = subprocess.run(
            ["markitdown", input_path],
            capture_output=True,
            text=True,
            check=True
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Ошибка конвертации: {e.stderr}")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    return FileResponse(
        output_path,
        media_type="text/markdown",
        filename="output.md"
    )