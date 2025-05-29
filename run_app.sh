"""
@author: Daniela Novoa, Orlando Arroyo, Frank Vidales
@owner: EstrucMed Ingeniería especializada S.A.S


"""  

#!/bin/bash

# Nombre del entorno
ENV_DIR=".venv"

echo "============================"
echo "→ Iniciando CSItoOpenSees"
echo "→ Verificando entorno..."
echo "============================"

# Crear entorno si no existe
if [ ! -d "$ENV_DIR" ]; then
    echo "→ Entorno virtual no encontrado. Creando..."
    python -m venv "$ENV_DIR"
    echo "→ Entorno creado en $ENV_DIR"
else
    echo "→ Entorno virtual ya existe."
fi

# Activar entorno (Windows Git Bash)
source "$ENV_DIR/Scripts/activate"

# Instalar dependencias
echo "→ Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

# Ejecutar aplicación
echo "→ Ejecutando script principal..."
python main.py


