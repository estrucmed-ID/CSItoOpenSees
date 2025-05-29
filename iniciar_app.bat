@echo off
setlocal

echo ================================
echo Iniciando la plataforma...
echo ================================

REM Verificar si ya existe el entorno virtual
if not exist ".venv" (
    echo [1/4] Creando entorno virtual .venv con Python 3.12...
    python -m venv .venv
)

REM Activar el entorno virtual
echo [2/4] Activando entorno virtual...
call .venv\Scripts\activate.bat

REM Instalar dependencias
echo [3/4] Instalando requerimientos...
pip install -r docs\requirements.txt > nul

REM Ejecutar la aplicación
echo [4/4] Ejecutando la plataforma...
python main.py

echo.
echo Presiona cualquier tecla para cerrar...
pause > nul

