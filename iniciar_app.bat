@echo off
echo -------------------------------
echo [1] Verificando entorno virtual
echo -------------------------------

if not exist ".venv" (
    echo Entorno virtual no encontrado. Creando entorno virtual con Python 3.12...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo ERROR: No se pudo crear el entorno virtual con Python 3.12. ¿Está instalado?
        pause
        exit /b 1
    )
)

echo ------------------------------------
echo [2] Activando entorno virtual .venv
echo ------------------------------------
call .venv\Scripts\activate.bat

echo -----------------------
echo [3] Actualizando pip...
echo -----------------------
python -m pip install --upgrade pip

echo ----------------------------------------
echo [4] Instalando dependencias (requirements)
echo ----------------------------------------
pip install -r docs\requirements.txt

echo ----------------------
echo [5] Ejecutando main.py
echo ----------------------
python main.py

if errorlevel 1 (
    echo ERROR: main.py encontró un problema. Abortando...
    pause
    exit /b 1
)

echo.
set /p VIEW_DASH="¿Desea visualizar el dashboard? (s/n): "

if /i "%VIEW_DASH%"=="s" (
    echo -------------------------------
    echo [6] Ejecutando generate_dashboard.py
    echo -------------------------------
    python dash_app\generate_dashboard.py
) else (
    echo Omitiendo dashboard...
)

pause


