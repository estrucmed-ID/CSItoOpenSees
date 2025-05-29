@echo off
set PYTHON_VERSION=3.12.10
set INSTALLER=python-%PYTHON_VERSION%-amd64.exe
set DOWNLOAD_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/%INSTALLER%

echo ----------------------------------------
echo [0] Verificando si Python %PYTHON_VERSION% está instalado...
echo ----------------------------------------

py -%PYTHON_VERSION% --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python %PYTHON_VERSION% no está instalado.
    echo Descargando instalador desde: %DOWNLOAD_URL%

    curl -o %INSTALLER% %DOWNLOAD_URL%
    if not exist %INSTALLER% (
        echo ERROR: No se pudo descargar Python. Verifique su conexión a internet.
        pause
        exit /b 1
    )

    echo Instalando Python %PYTHON_VERSION% silenciosamente...
    %INSTALLER% /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

    if %errorlevel% neq 0 (
        echo ERROR: Falló la instalación de Python.
        pause
        exit /b 1
    )

    echo Python instalado correctamente.
)

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

pause


