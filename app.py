import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from markitdown import MarkItDown

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Отдаём статику
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/convert")
async def convert_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Только PDF-файлы!")

    original_name = file.filename
    safe_base_name = os.path.splitext(original_name)[0]
    unique_id = str(uuid.uuid4())[:8]

    input_path = os.path.join(UPLOAD_DIR, f"{unique_id}.pdf")
    output_path = os.path.join(OUTPUT_DIR, f"{safe_base_name}.md")

    # Сохраняем PDF
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Конвертируем через библиотеку markitdown (без subprocess!)
    try:
        markitdown = MarkItDown()
        result = markitdown.convert(input_path)
        markdown_text = result.text_content

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка конвертации: {str(e)}")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    return FileResponse(
        output_path,
        media_type="text/markdown; charset=utf-8",
        filename=f"{safe_base_name}.md"
    )