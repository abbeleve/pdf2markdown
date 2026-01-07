@echo off
chcp 65001 >nul
setlocal

echo === PDF to Markdown Converter ===
echo.

:: Используем 'py' вместо 'python' — надёжнее на Windows
where py >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add to PATH" during installation.
    echo.
    pause
    exit /b 1
)

if not exist "venv" (
    echo [1/3] Creating virtual environment...
    py -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
)

call venv\Scripts\activate

echo [2/3] Installing packages...
venv\Scripts\pip.exe install markitdown[all] fastapi uvicorn python-multipart --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install packages.
    pause
    exit /b 1
)

if not exist "app.py" (
    echo [ERROR] app.py not found!
    pause
    exit /b 1
)

echo [3/3] Launching server at http://127.0.0.1:8000
start "" "http://127.0.0.1:8000"
venv\Scripts\uvicorn.exe app:app --host 127.0.0.1 --port 8000

if errorlevel 1 (
    echo.
    echo [ERROR] Server failed to start.
)

echo.
echo Server stopped.
pause