@echo off
echo Activando entorno virtual...
call .venv\Scripts\activate.bat

echo Ejecutando la aplicación...
python main.py

pause
