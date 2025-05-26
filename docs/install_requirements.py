"""
@author: Daniela Novoa, Orlando Arroyo, Frank Vidales
@owner: EstrucMed Ingeniería especializada S.A.S


"""  

# install_requirements.py

import subprocess
import sys
from tqdm import tqdm

#%% Install packages

def install_packages(requirements_file='requirements.txt'):
    try:
        with open(requirements_file, 'r') as f:
            packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo {requirements_file}")
        sys.exit(1)

    print("\nInstalando paquetes...\n")
    for package in tqdm(packages, desc="Instalando", ncols=80):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"\nError instalando {package}: {e}")
    
    print("\n✅ Instalación completada.\n")

if __name__ == "__main__":
    install_packages()
