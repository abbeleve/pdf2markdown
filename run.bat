@echo off
chcp 65001 >nul
setlocal

echo === PDF to Markdown Converter ===
echo.

:: Проверка Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    echo Install Python 3.9+ from https://www.python.org/downloads/
    echo Check "Add to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: Создаём venv, если нет
if not exist "venv" (
    echo [1/3] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
)

:: Активируем
call venv\Scripts\activate

:: Устанавливаем зависимости
echo [2/3] Installing markitdown and web server...
venv\Scripts\pip.exe install markitdown fastapi uvicorn python-multipart --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install packages.
    pause
    exit /b 1
)

:: Проверяем app.py
if not exist "app.py" (
    echo [ERROR] app.py not found in this folder!
    pause
    exit /b 1
)

:: Запуск
echo [3/3] Launching server at http://127.0.0.1:8000
start "" "http://127.0.0.1:8000"
venv\Scripts\uvicorn.exe app:app --host 127.0.0.1 --port 8000

if errorlevel 1 (
    echo.
    echo [ERROR] Server failed to start.
    echo Is port 8000 already in use?
)

echo.
echo Server stopped.
pause